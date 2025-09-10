from typing import Optional, Tuple, List, Dict
import json
import logging

from schemas.state import AgentState
from utils.logger import get_logger, log_execution

logger = get_logger(__name__)


class AlignerAgent:
    """对齐者（基于多模态大模型的实现）

    主要职责：使用可配置的多模态大模型（MM-LLM）来
      1. 从辩论者输出中抽取与模态相关的断言（claims）;
      2. 将这些断言与输入的模态（如图像/视频/音频）一并提交给多模态大模型进行逐条校验;
      3. 根据模型返回结果更新 state，并返回 (aligned: bool, feedback: Optional[str])。

    设计要点：
    - 优先使用 settings 提供的多模态模型实例 (settings.get_multimodal_llm())；
      该实例应支持直接接收 text + images 并返回结构化 JSON 的能力（详见下方 "期望的 MM-LLM 接口"）。
    - 若未提供专门的 MM-LLM，则降级为单纯的文本解析或提醒用户提供视觉后端/检测结果。
    - 不改变工作流控制逻辑（仅通过返回值与 state 字段影响下一步）。
    """

    def __init__(self):
        from config import settings

        # 优先尝试取多模态大模型
        self.mm = None
        try:
            self.mm = settings.get_multimodal_llm()
            logger.info("对齐者已加载多模态大模型：%s", getattr(self.mm, "model_name", "<multimodal_model>"))
        except Exception:
            logger.warning("未能从 settings 获取多模态模型，尝试退回到普通 LLM 或启发式方法")
            try:
                self.llm = settings.get_llm("aligner")
                logger.info("对齐者已加载普通 LLM：%s", getattr(self.llm, "model_name", "<llm>"))
            except Exception:
                self.llm = None
                logger.warning("未能加载任何 LLM 实例，Aligner 将功能受限")

    @log_execution()
    def check_alignment(self, state: AgentState) -> Tuple[bool, Optional[str]]:
        """主入口：对齐检查

        返回： (aligned: bool, feedback: Optional[str])
        - aligned True: 全部断言通过
        - aligned False: feedback 为可读文本，列出未通过的断言与原因，供辩论者使用
        """
        logger.info("开始多模态对齐检查")

        # 从 state 提取辩论者输出与输入模态
        claims_text = self._gather_debate_text(state)
        images = state.get("images") or state.get("input_images") or []
        videos = state.get("videos") or []
        audios = state.get("audios") or []

        if not claims_text.strip():
            logger.info("未检测到辩论文本，跳过对齐")
            return True, None

        # 使用 MM-LLM 执行断言抽取 + 对齐校验
        if self.mm is not None:
            try:
                # 期望 mm 支持一个 verify_alignment 接口：接收文本 + 模态资源并返回结构化 JSON
                if hasattr(self.mm, "verify_alignment"):
                    logger.info("调用 mm.verify_alignment 进行校验")
                    resp = self.mm.verify_alignment(
                        text=claims_text,
                        images=images,
                        videos=videos,
                        audios=audios,
                        return_json=True,
                    )
                    return self._parse_mm_response(resp, state)

                # 否则，调用通用 chat/generate 接口并通过 prompt 指导其返回 JSON
                if hasattr(self.mm, "chat"):
                    prompt = self._build_prompt_for_mm(claims_text)
                    logger.info("调用 mm.chat（降级路径），通过 prompt 让模型进行校验并以 JSON 返回结果")
                    mm_resp = self.mm.chat(prompt, images=images)
                    # mm_resp 可能为字符串（含 JSON），尝试解析
                    return self._parse_mm_response(mm_resp, state)

                # 若 mm 仅有 generate 接口
                if hasattr(self.mm, "generate"):
                    prompt = self._build_prompt_for_mm(claims_text)
                    mm_resp = self.mm.generate(prompt, attachments=images)
                    return self._parse_mm_response(mm_resp, state)

            except Exception as e:
                logger.exception("调用多模态模型进行对齐校验时失败: %s", e)
                # 降级 -> 尝试使用普通 LLM 或返回不可校验消息

        # 如果没有多模态模型，但有普通 LLM，我们尝试提取断言并告知需要视觉后端
        if getattr(self, "llm", None) is not None:
            try:
                logger.info("使用普通 LLM 抽取断言，但无法完成模态验证，返回需要视觉后端的提示")
                assertions = self._extract_assertions_with_llm(claims_text)
                feedback_items = []
                for a in assertions:
                    feedback_items.append(f"需要校验：{a.get('text')}（请提供视觉/音频检测结果或配置多模态模型）")
                feedback = "；".join(feedback_items) if feedback_items else None
                if feedback:
                    state["alignment_details"] = {"assertions": assertions}
                    return False, feedback
                return True, None
            except Exception:
                logger.exception("使用普通 LLM 抽取断言失败")

        # 最后退回：既无 MM-LLM 也无 LLM -> 不能校验
        logger.warning("未配置多模态模型或视觉后端，无法进行对齐校验")
        return False, "无法进行对齐校验：未配置多模态模型或未提供模态检测结果。"

    # ---------- 辅助函数 ----------
    def _gather_debate_text(self, state: AgentState) -> str:
        """从 state 中聚合辩论者产生的文本，作为多模态模型的输入文本部分。"""
        parts = []
        for k in ("debate_output", "debate_claims", "last_debate_message", "debate_messages"):
            v = state.get(k)
            if not v:
                continue
            if isinstance(v, list):
                parts.extend(map(str, v))
            else:
                parts.append(str(v))
        return "".join(parts)

    def _build_prompt_for_mm(self, claims_text: str) -> str:
        """构建 prompt，要求 MM-LLM 对每条断言进行与给定模态的逐条校验，并以严格的 JSON 格式返回结果。

        返回 JSON 结构示例：
        {
          "aligned": false,
          "failures": [
            {"claim": "图像中有黄色房子", "reason": "未在图像中检测到黄色房子，检测到房子为白色", "confidence": 0.12},
            ...
          ],
          "details": [
            {"claim": ..., "evidence": "第1张图像左下角的区域没有黄色像素"}
          ]
        }
        """
        prompt = (
                        "请检查下列辩论者的断言是否与随消息一起提供的模态输入（图像/视频/音频）一致。"
                        "对于每一条断言，逐条判断是否在模态中能够被证实。如果断言无法证实或被证伪，提供失败原因和简要证据。"
                        '''输出必须是有效的 JSON，格式如下：{ 
            \"aligned\": bool,
            \"failures\": [{\"claim\": str, \"reason\": str, \"confidence\": float}],
            \"details\": [...]
}'''
            "被检查的断言如下:" + claims_text + "请只以 JSON 返回，不要包含多余的文字。"
)
        return prompt

    def _parse_mm_response(self, mm_resp, state: AgentState) -> Tuple[bool, Optional[str]]:
        """解析多模态模型的响应，更新 state 并返回 (aligned, feedback)。

        mm_resp 可以是：
          - 直接返回的 dict（已经是结构化数据），或
          - 包含 JSON 的字符串（需要解析）。
        """
        parsed = None
        try:
            if isinstance(mm_resp, dict):
                parsed = mm_resp
            else:
                # 尝试从字符串中提取首个 JSON 对象
                if isinstance(mm_resp, str):
                    parsed = json.loads(mm_resp)
                else:
                    # 某些 SDK 返回复杂对象，尝试取 text 字段
                    if hasattr(mm_resp, "text"):
                        parsed = json.loads(mm_resp.text)
                    elif hasattr(mm_resp, "content"):
                        parsed = json.loads(mm_resp.content)
                    else:
                        # 最后退化为尝试 str()
                        parsed = json.loads(str(mm_resp))
        except Exception as e:
            logger.exception("解析多模态模型返回的 JSON 失败: %s", e)
            # 尝试做简单容错：从字符串中查找 { 和 } 并解析
            try:
                s = mm_resp if isinstance(mm_resp, str) else getattr(mm_resp, "text", str(mm_resp))
                start = s.find("{")
                end = s.rfind("}")
                if start != -1 and end != -1 and start < end:
                    parsed = json.loads(s[start:end+1])
            except Exception:
                logger.exception("二次尝试解析 MM 返回的 JSON 失败")

        if not parsed:
            logger.warning("多模态模型没有返回可解析的结构化结果，无法确定对齐情况")
            return False, "多模态模型返回无法解析的结果。请确保模型以 JSON 格式返回对齐校验结果。"

        # 解析标准字段
        aligned = bool(parsed.get("aligned", False))
        failures = parsed.get("failures") or []
        details = parsed.get("details") or parsed.get("evidence") or []

        # 把解析结果写入 state，便于后续分析或仲裁使用
        state["alignment_passed"] = aligned
        state["alignment_details"] = {
            "failures": failures,
            "details": details,
            "raw": parsed,
        }

        if not aligned:
            # 构建用户友好的反馈文本（可直接输回给辩论者，让其据此纠正）
            failure_texts = []
            for f in failures:
                c = f.get("claim") or f.get("text") or str(f)
                r = f.get("reason") or "未被模态证实"
                confidence = f.get("confidence")
                if confidence is not None:
                    failure_texts.append(f"{c} -> {r} (confidence={confidence})")
                else:
                    failure_texts.append(f"{c} -> {r}")

            feedback = "；".join(failure_texts)
            state["alignment_feedback"] = feedback
            logger.info("对齐检查未通过，反馈已写入 state")
            return False, feedback

        logger.info("对齐检查通过")
        state.pop("alignment_feedback", None)
        return True, None

    def _extract_assertions_with_llm(self, claims_text: str) -> List[Dict]:
        """在没有 MM-LLM 的情况下使用普通 LLM 抽取断言（仅用于提示用户）。"""
        if not getattr(self, "llm", None):
            return []
        try:
            if hasattr(self.llm, "parse_assertions"):
                return self.llm.parse_assertions(claims_text)
            # 否则用简单 prompt 请求模型返回 JSON 数组
            prompt = (
                "请从下面的辩论文本中提取所有涉及非文本模态（图片/视频/音频）的断言，"
                "以 JSON 数组返回，每个元素包含字段 {\"text\":..., \"predicate\":... , \"target\":..., \"value\":...}"
                + claims_text
            )
            resp = self.llm.chat(prompt)
            if isinstance(resp, str):
                return json.loads(resp)
            if isinstance(resp, dict) and resp.get("choices"):
                # some llm SDK 返回格式
                text = resp["choices"][0]["message"]["content"]
                return json.loads(text)
            return []
        except Exception:
            logger.exception("使用普通 LLM 抽取断言失败")
            return []

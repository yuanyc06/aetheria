# agents/supporter.py
from schemas.state import AgentState
from tools.tool_pool import tool_pool
from utils.logger import get_logger, log_execution
from config import settings
import wikipedia
from typing import List
import time
import json

logger = get_logger(__name__)

# class SupporterAgent:
#     def __init__(self):
#         from config import settings
#         self.llm = settings.get_llm("supporter")
#         logger.info("支持者智能体已初始化，使用模型: %s", self.llm.model_name)

#     @log_execution()
#     def collect_background(self, state: AgentState) -> dict:
#         """支持者节点 - 收集背景信息"""
#         logger.info("开始收集背景信息")
        
#         # 根据内容获取相关安全政策背景
#         prompt = settings.PROMPT_TEMPLATES["collect_background"].format(
#             translated_text=state["translated_text"]
#         )
        
#         logger.debug("背景收集提示: %s", prompt[:200] + "...")
        
#         response = self.llm.invoke(prompt)
#         background = response.content
        
#         # 添加工具提供的额外背景信息
#         additional_background = self._gather_additional_info(state)
#         background += "\n\n额外背景信息:\n" + additional_background
        
#         logger.info("背景信息收集完成 (长度: %d)", len(background))
#         logger.debug("背景摘要: %s", background[:150] + "...")
        
#         return {
#             "background": background,
#             "status": "background_collected"
#         }
    
#     def _gather_additional_info(self, state: AgentState) -> str:
#         """使用工具收集额外背景信息"""
#         additional_info = []
        
#         # 如果包含文本，使用文本分析工具获取更多上下文
#         if "text" in state.get("modalities", []):
#             text_content = state["raw_input"].get("text", "")
#             if text_content:
#                 text_analysis = tool_pool.execute_tool(
#                     "text_safety_check", 
#                     text_content,
#                     "需要更多上下文背景"
#                 )
#                 additional_info.append(f"文本深度分析:\n{text_analysis}")
        
#         # 如果包含图像，可以添加图像特定分析
#         if "image" in state.get("modalities", []):
#             # 这里可以添加特定于图像的背景收集
#             additional_info.append("图像背景: 考虑图像内容的文化象征意义和潜在隐喻")
        
#         return "\n".join(additional_info)


class SupporterAgent:
    def __init__(self):
        self.llm = settings.get_llm("supporter")
        logger.info("支持者智能体已初始化，使用模型: %s", self.llm.model_name)

    @log_execution()
    def collect_background(self, state: AgentState) -> dict:
        """支持者节点 - 收集背景信息
        """
        logger.info("开始收集背景信息")

        # 首先从 settings 中读取新的模板，如果没有则使用内置默认模板。
        decision_template = settings.PROMPT_TEMPLATES.get(
    "collect_background_v2",
    """你是内容安全背景收集助手。
请判断下面的用户输入是否需要额外的背景检索来做内容安全评估,你只需要关注自己无法理解的实体或关键词。
若不需要，请返回 JSON：{{"need_background": false, "explanation": "简短说明"}}
若需要，请返回 JSON：{{"need_background": true, "keywords": ["实体或关键词1","实体2"], "search_focus": "检索时重点关注的信息（例如：人物背景、历史争议、敏感事件等）"}}
**只返回纯 JSON，不要添加其它说明或多余文本。**
文本如下：
{translated_text}
"""
)

        # 格式化时只提供 translated_text 参数
        decision_prompt = decision_template.format(translated_text=state["translated_text"])
        logger.debug("背景决策提示: %s", (decision_prompt[:200] + "...") if len(decision_prompt) > 200 else decision_prompt)

        decision_resp = self.llm.invoke(decision_prompt)
        decision_raw = decision_resp.content.strip()

        # 解析模型的 JSON 输出（容错处理）
        need_background = True
        keywords = []
        search_focus = None
        try:
            decision = json.loads(decision_raw)
            need_background = bool(decision.get("need_background", True))
            keywords = decision.get("keywords") or []
            search_focus = decision.get("search_focus") or decision.get("explanation") or None
        except Exception:
            # 若解析失败，尝试降级解析并回退到关键词抽取
            logger.debug("解析决策结果为 JSON 失败，原始输出: %s", decision_raw)
            # 如果输出明显表明无需检索，则尊重模型判断
            if "need_background" in decision_raw and "false" in decision_raw.lower():
                need_background = False
            else:
                # 否则默认认为需要背景，并从文本中抽取关键词（降级方案）
                keywords = self._extract_search_terms(state["translated_text"], max_terms=5)

        if not need_background:
            # 模型判断无需额外检索，使用模型返回的解释作为 background（如果有）
            try:
                decision = json.loads(decision_raw)
                explanation = decision.get("explanation")
            except Exception:
                explanation = decision_raw
            background = explanation or "(模型判断无需检索背景)"
        else:
            # 确保有关键词；若为空则用已有的抽取函数补全
            if not keywords:
                keywords = self._extract_search_terms(state["translated_text"], max_terms=2)
            logger.info("需要检索，关键词：%s", keywords)

            # 针对每个关键词查询维基并拼接摘要
            wiki_summaries = []
            for t in keywords:
                if not t:
                    continue
                try:
                    summary = self._get_wikipedia_summary(t, sentences=5)
                    if summary:
                        wiki_summaries.append(f"词: {t}{summary}")
                except Exception as e:
                    logger.debug("维基查询失败（%s）：%s", t, e)
                time.sleep(0.2)

            wiki_block = "".join(wiki_summaries) if wiki_summaries else "(未检索到维基摘要)"

            # 让模型基于原文、维基摘要与额外信息做最终的总结整理（可被 settings 覆盖）
            summarize_template = settings.PROMPT_TEMPLATES.get(
                "summarize_background",
                """你是内容安全总结助手。请基于下面的用户文本、维基检索结果，写一份面向内容安全评估的背景汇总，注意，仅对维基百科检索结果的有用信息进行总结。
输出可以包含：
1) 简洁背景汇总（2-4 段）;
2) 关键信息要点（以短句或项目符列出）;
3) 潜在风险点（列出）;
如果没有维基百科检索结果，则直接输出 无背景信息。

用户文本：
{translated_text}

维基检索结果：
{wiki_summaries}
"""
            )

            summarize_prompt = summarize_template.format(
                translated_text=state["translated_text"],
                wiki_summaries=wiki_block,
            )
            logger.debug("汇总提示: %s", (summarize_prompt[:200] + "...") if len(summarize_prompt) > 200 else summarize_prompt)

            summary_resp = self.llm.invoke(summarize_prompt)
            background = summary_resp.content

        return {
            "background": background,
            "status": "background_collected",
            "wiki_summaries": wiki_block,
        }

    def _get_wikipedia_summary(self, term: str, sentences: int = 3) -> str:
        """安全地从维基获取摘要，处理消歧义和未找到的情况。"""
        try:
            return wikipedia.summary(term, sentences=sentences)
        except wikipedia.DisambiguationError as e:
            # 选择可能性中的第一个并重试（保守策略）
            logger.debug("消歧义: %s -> 选取第一个候选 %s", term, e.options[:3])
            try:
                return wikipedia.summary(e.options[0], sentences=sentences)
            except Exception:
                return "(维基存在歧义，未能自动选择摘要)"
        except wikipedia.PageError:
            logger.debug("维基无此页: %s", term)
            return "(未能在维基百科找到匹配条目)"
        except Exception as e:
            logger.debug("维基获取摘要出错: %s", e)
            return "(维基查询出错)"

    def _extract_search_terms(self, text: str, max_terms: int = 5) -> List[str]:
        """使用 LLM 从文本中抽取若干供搜索的关键词/实体。

        优点：适配中英文、短语、实体名，与简单的正则或分词比更稳健。
        返回：去重后的关键词列表（最多 max_terms 个）。
        """
        if not text or not text.strip():
            return []

        # 让模型返回一个用逗号分隔的关键词列表
        extract_prompt = (
            "请从下面的文本中抽取最多 {n} 个用于检索的关键词或实体，返回时用逗号分隔。"
            "不要添加额外说明，只返回关键词列表。\n\n文本:\n{txt}"
        ).format(n=max_terms, txt=text)

        try:
            resp = self.llm.invoke(extract_prompt)
            raw = resp.content.strip()
            # 规范化：用逗号或换行拆分，并去重
            parts = [p.strip() for p in raw.replace("\n", ",").split(",") if p.strip()]
            seen = set()
            terms = []
            for p in parts:
                if p not in seen:
                    seen.add(p)
                    terms.append(p)
                if len(terms) >= max_terms:
                    break
            return terms
        except Exception as e:
            logger.debug("LLM 抽取关键词失败: %s", e)
            return []

    def _gather_additional_info(self, state: AgentState) -> str:
        """使用工具收集额外背景信息（保持原有逻辑并补充维基查询结果）。"""
        additional_info = []

        # 如果包含文本，使用文本分析工具获取更多上下文
        if "text" in state.get("modalities", []):
            text_content = state["raw_input"].get("text", "")
            if text_content:
                try:
                    text_analysis = tool_pool.execute_tool(
                        "text_safety_check",
                        text_content,
                        "需要更多上下文背景"
                    )
                    additional_info.append(f"文本深度分析:\n{text_analysis}")
                except Exception as e:
                    logger.debug("调用 text_safety_check 失败: %s", e)

        # 如果包含图像，可以添加图像特定分析
        if "image" in state.get("modalities", []):
            # 这里可以添加特定于图像的背景收集，例如调用图像识别工具
            additional_info.append("图像背景: 考虑图像内容的文化象征意义和潜在隐喻")

        return "\n".join(additional_info)



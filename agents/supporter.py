from schemas.state import AgentState
from tools.tool_pool import tool_pool
from utils.logger import get_logger, log_execution
from config import settings
from typing import List
import time
import json
from baidusearch.baidusearch import search as baidu_search
from tools.baidu_image_search import search_image_urls
from tools.rag_tool import rag_tool

logger = get_logger(__name__)


class SupporterAgent:
    def __init__(self):
        self.llm = settings.get_llm("supporter")
        logger.info("支持者智能体已初始化，使用模型: %s", self.llm.model_name)

    @log_execution()
    def collect_background(self, state: AgentState) -> dict:
        """支持者节点 - 收集背景信息"""
        logger.info("开始收集背景信息")

        # 首先从 settings 中读取新的模板，如果没有则使用内置默认模板。
        decision_template = settings.PROMPT_TEMPLATES.get(
            "collect_background")

        # 格式化时只提供 translated_text 参数
        decision_prompt = decision_template.format(
            translated_text=state["translated_text"]
        )
        logger.debug(
            "背景决策提示: %s",
            (
                (decision_prompt[:200] + "...")
                if len(decision_prompt) > 200
                else decision_prompt
            ),
        )

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
            search_focus = (
                decision.get("search_focus") or decision.get("explanation") or None
            )
        except Exception:
            # 若解析失败，尝试降级解析并回退到关键词抽取
            logger.debug("解析决策结果为 JSON 失败，原始输出: %s", decision_raw)
            # 如果输出明显表明无需检索，则尊重模型判断
            if "need_background" in decision_raw and "false" in decision_raw.lower():
                need_background = False
            else:
                # 否则默认认为需要背景，并从文本中抽取关键词（降级方案）
                keywords = self._extract_search_terms(
                    state["translated_text"], max_terms=5
                )

        # 预先定义用于返回的检索结果块，兼容原有字段名
        web_block = ""
        image_block = ""
        need_background = True
        if not need_background:
            # 模型判断无需额外检索，使用模型返回的解释作为 background（如果有）
            try:
                decision = json.loads(decision_raw)
                explanation = decision.get("explanation")
            except Exception:
                explanation = decision_raw
            background = explanation or "(No background retrieval required)"
        else:
            if not keywords:
                keywords = self._extract_search_terms(
                    state["translated_text"], max_terms=5
                )
            logger.info("需要检索，关键词：%s", keywords)

            web_summaries = []
            image_summaries = []
            for t in keywords[:2]:
                if not t:
                    continue
                try:
                    results = self._search_baidu(t, max_results=1)
                    if results:
                        for r in results:
                            title = r.get("title", "")
                            abstract = r.get("abstract", "")
                            url = r.get("url", "")
                            web_summaries.append(
                                f"word: {t} | title: {title} | abstract: {abstract} \n"
                            )
                except Exception as e:
                    logger.debug("百度搜索失败（%s）：%s", t, e)
                time.sleep(0.2)

            web_block = (
                "".join(web_summaries) if web_summaries else "(No Baidu search results found.)"
            )

            # 图片检索：当输入包含图像模态并提供本地图片路径时执行“以图搜图”
            image_block = "(No image results retrieved)"
            try:
                if "image" in state.get("modalities", []):
                    raw = state.get("raw_input", {}) or {}
                    # 兼容多种可能字段名
                    candidate_paths = [
                        raw.get("image_path"),
                        raw.get("image_file"),
                        raw.get("image"),
                    ]
                    image_path = next(
                        (
                            p
                            for p in candidate_paths
                            if isinstance(p, str) and p.strip()
                        ),
                        None,
                    )
                    if image_path:
                        from pathlib import Path as _Path

                        try:
                            urls = search_image_urls(_Path(image_path), max_results=5)
                        except Exception as e:
                            logger.debug("以图搜图失败（%s）：%s", image_path, e)
                            urls = []
                        if urls:
                            for u in urls:
                                image_summaries.append(f"图片: {u}\n")
                            image_block = "".join(image_summaries)
            except Exception as e:
                logger.debug("图片检索流程出错: %s", e)

            # RAG 历史案例搜索
            historical_cases = ""
            try:
                # 显示案例库状态
                import os

                reports_dir = "reports"
                if os.path.exists(reports_dir):
                    report_files = [
                        f for f in os.listdir(reports_dir) if f.endswith(".txt")
                    ]
                    # print(f"📁 案例库状态: 存在 {len(report_files)} 个历史报告文件")
                else:
                    print("📁不存在")

                logger.info("开始搜索历史案例库...")
                print("🔍 开始搜索历史案例库...")
                historical_cases = rag_tool.search_historical_cases(
                    state["translated_text"], max_results=3
                )
                if (
                    historical_cases
                    and historical_cases != "RAG 系统未初始化，无法搜索历史案例"
                    and "No relevant historical cases found." not in historical_cases
                ):
                    logger.info("✅ 从案例库中搜索到相关历史案例")
                    print("✅ 从案例库中搜索到相关历史案例")
                    # 显示案例摘要
                    lines = historical_cases.split("\n")
                    print("相关历史案例摘要:")
                    for line in lines[:3]:  # 只显示前3行
                        if line.strip():
                            print(f"  {line.strip()}")
                else:
                    logger.info("❌ 未从案例库中搜索到相关历史案例")
                    print("❌ 未从案例库中搜索到相关历史案例")
                    historical_cases = "(No relevant historical cases found.)"
            except Exception as e:
                logger.debug("RAG 历史案例搜索失败: %s", e)
                print(f"⚠️ 历史案例搜索出错: {e}")
                historical_cases = "(No relevant historical cases found.)"

            # 让模型基于原文、摘要与额外信息做最终的总结整理（可被 settings 覆盖）
            summarize_template = settings.PROMPT_TEMPLATES.get(
                "summarize_background",
            )

            summarize_prompt = summarize_template.format(
                translated_text=state["translated_text"],
                wiki_summaries=web_block,
                image_summaries=image_block,
                historical_cases=historical_cases,
            )
            logger.debug(
                "汇总提示: %s",
                (
                    (summarize_prompt[:200] + "...")
                    if len(summarize_prompt) > 200
                    else summarize_prompt
                ),
            )

            summary_resp = self.llm.invoke(summarize_prompt)
            background = summary_resp.content

        return {
            "background": background,
            "status": "background_collected",
            "wiki_summaries": web_block,
            # "image_summaries": image_block,
        }

    def _search_baidu(self, term: str, max_results: int = 3) -> List[dict]:
        """使用 baidusearch 进行关键词检索，返回若干条结果。"""
        try:
            results = baidu_search(term)
            if not isinstance(results, list):
                return []
            # 仅保留 title/abstract/url 字段，并裁剪数量
            cleaned = []
            for item in results[: max_results if max_results > 0 else None]:
                if not isinstance(item, dict):
                    continue
                cleaned.append(
                    {
                        "title": item.get("title", ""),
                        "abstract": item.get("abstract", ""),
                        "url": item.get("url", ""),
                    }
                )
            return cleaned
        except Exception as e:
            logger.debug("调用 baidusearch 出错: %s", e)
            return []

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
        """使用工具收集额外背景信息。"""
        additional_info = []

        # 如果包含文本，使用文本分析工具获取更多上下文
        if "text" in state.get("modalities", []):
            text_content = state["raw_input"].get("text", "")
            if text_content:
                try:
                    text_analysis = tool_pool.execute_tool(
                        "text_safety_check", text_content, "需要更多上下文背景"
                    )
                    additional_info.append(f"文本深度分析:\n{text_analysis}")
                except Exception as e:
                    logger.debug("调用 text_safety_check 失败: %s", e)

        # 如果包含图像，可以添加图像特定分析
        if "image" in state.get("modalities", []):
            # 这里可以添加特定于图像的背景收集，例如调用图像识别工具
            additional_info.append("图像背景: 考虑图像内容的文化象征意义和潜在隐喻")

        return "\n".join(additional_info)

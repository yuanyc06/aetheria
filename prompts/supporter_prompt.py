collect_background_zh = """你是内容安全信息识别助手。
请判断下面的用户输入是否有对于内容安全判别至关重要但你无法理解的实体或名词。
若没有，请返回 JSON：{{"need_background": false, "explanation": "简短说明"}}
若有，请返回 JSON：{{"need_background": true, "keywords": ["实体或关键词1","实体2"], "search_focus": "检索时重点关注的信息（例如：人物背景、历史争议、敏感事件等）"}}
**只返回纯 JSON，不要添加其它说明或多余文本。**
用户输入（若包含非文本模态，已转为文本描述）：
{translated_text}"""

collect_background_en = """You are a content safety information identification assistant.
Please determine whether the following user input contains entities or terms that are crucial for content safety judgment but that you cannot understand.
If not, please return JSON: {{"need_background": false, "explanation": "brief explanation"}}
If yes, please return JSON: {{"need_background": true, "keywords": ["entity_or_keyword1","entity2"], "search_focus": "key information to focus on during retrieval (e.g., background of the person, historical controversies, sensitive events, etc.)"}}
**Return pure JSON only, without adding any other explanation or extra text.**
User input (if it contains non-text modalities, they have been converted into text descriptions):
{translated_text}"""



summarize_background_zh = """你是内容安全总结助手。请基于下面的用户文本、百度检索结果、图片检索结果与历史案例，写一份面向内容安全评估的背景汇总，注意，仅对检索结果的有用信息进行总结。
输出可以包含：
1) 关键背景汇总（2-4 段）;
2) 相关历史案例参考（如有）;
如果没有检索结果，则直接输出 无背景信息。

用户文本：
{translated_text}

百度检索结果：
{wiki_summaries}

图片检索结果：
{image_summaries}

历史案例：
{historical_cases}
"""

summarize_background_en = """You are a content safety summarization assistant. 
Based on the following user text, Baidu search results, image search results, and historical cases, 
write a background summary oriented to content safety evaluation. 
Note: Only summarize useful information from the retrieved results.

Your output may include:
1) Key background summary (2–4 paragraphs);
2) References to relevant historical cases (if any);
If there are no retrieval results, simply output: No background information.

User text:
{translated_text}

Baidu search results:
{wiki_summaries}

Image search results:
{image_summaries}

Historical cases:
{historical_cases}
"""


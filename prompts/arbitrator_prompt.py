#  you should pay attention to both implicit and explicit risks.
arbitrator_en = """As a content safety arbitrator, 
 Please generate the final risk assessment report based on the debaters' debate history:

User Instruction:
{instruction}

User Input:
{translated_text}

Background Analysis:
{background}

Debate History:
{debate_summary}

Report Requirements:
Based on the user input and the viewpoints of both debaters, arbitrate the debate outcome and indicate whether a content safety risk exists (Yes/No). 
Summarize the reasoning process from the debate history that supports the final conclusion.

## Output Format:
## Risk Exists: [Yes/No]
## Reasoning:
[Detailed summary of the reasoning process in the debate history] (200 words)

"""


arbitrator_zh = """作为一名内容安全仲裁员，能够关注隐性风险和显性风险，请根据辩论者的辩论历史生成最终的风险评估报告：


用户指令：
{instruction}

用户输入：
{translated_text}

背景分析：
{background}

辩论历史：
{debate_summary}

报告要求：
基于用户输入和辩论双方的观点，对辩论结果进行仲裁，说明是否存在内容安全风险（是/否），
总结辩论历史中支持最终观点的推理过程，

## 输出格式：
## 是否存在风险: [是/否]
## 推理过程:
[辩论历史中推理过程的详细总结]（200字）

"""

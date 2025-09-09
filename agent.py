import os
from typing import Annotated, Any, Dict, List, Literal, TypedDict
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
import base64
import requests

# # 环境变量设置（请替换为您的实际API密钥）
openkey_api_key = os.environ["OPENKEY_API_KEY"]
# tavily_api_key = os.environ["TAVILY_API_KEY"]


# 1. 定义状态结构
class AgentState(TypedDict):
    # 输入数据
    instruction: str
    text: str
    image: Annotated[str, "Base64编码的图像数据"]
    
    # 处理过程数据
    image_description: Annotated[str, "图像的详细文字描述"]
    search_results: Annotated[str, "网络搜索获取的相关知识"]
    debate_rounds: Annotated[List[str], "辩论记录"]
    final_summary: Annotated[str, "最终总结结果"]

# 2. 初始化关键组件
# 多模态模型（用于图像描述）
vision_model = ChatOpenAI(model="doubao-1.5-vision-pro-250328", max_tokens=1024,api_key=openkey_api_key, base_url="https://openkey.cloud/v1")
# 规划与总结模型
planner_model = ChatOpenAI(model="gpt-4o",api_key=openkey_api_key, base_url="https://openkey.cloud/v1")
# 辩论模型
debater_model = ChatOpenAI(model="qwen3-235b-a22b",api_key=openkey_api_key, base_url="https://openkey.cloud/v1")
# 网络搜索工具
search_tool = TavilySearchResults(max_results=3)

# 3. 定义节点函数
def preprocess_node(state: AgentState) -> Dict[str, Any]:
    """预处理节点：理解指令并生成图像描述"""
    # 生成图像描述
    if state["image"]:
        image_url = f"data:image/jpeg;base64,{state['image']}"
        msg = vision_model.invoke(
            [
                AIMessage(content="你是一个专业的图像分析师，请详细描述图像内容，注意所有细节："),
                HumanMessage(content=[
                    {"type": "text", "text": state["instruction"]},
                    {"type": "image_url", "image_url": {"url": image_url}}
                ])
            ]
        )
        image_description = msg.content
    else:
        image_description = "无图像输入"
    
    return {
        "image_description": image_description,
        "debate_rounds": []  # 初始化辩论记录
    }

def plan_node(state: AgentState) -> Dict[str, Any]:
    """规划节点：确定处理步骤"""
    response = planner_model.invoke(
        f"## 任务说明\n{state['instruction']}\n\n"
        f"## 文本内容\n{state['text']}\n\n"
        f"## 图像描述\n{state['image_description']}\n\n"
        "请规划处理步骤，首先需要搜索相关知识，然后进行三轮辩论，最后总结结果。"
        "只需回复'开始'即可。"
    )
    return {"plan": response.content}

def search_node(state: AgentState) -> Dict[str, Any]:
    """知识搜索节点：获取背景信息"""
    query = (
        f"搜索与以下内容相关的安全知识："
        f"文本: {state['text'][:300]}... "
        f"图像描述: {state['image_description'][:300]}..."
    )
    results = search_tool.invoke({"query": query})
    search_content = "\n\n".join([
        f"来源: {res['url']}\n内容: {res['content']}" 
        for res in results
    ])
    return {"search_results": search_content}

def debate_node(state: AgentState) -> Dict[str, Any]:
    """辩论节点：正反双方辩论"""
    # 构建辩论上下文
    context = (
        f"### 任务指令\n{state['instruction']}\n\n"
        f"### 文本内容\n{state['text']}\n\n"
        f"### 图像描述\n{state['image_description']}\n\n"
        f"### 背景知识\n{state['search_results']}\n\n"
    )
    
    # 确定当前辩论轮次
    round_num = len(state["debate_rounds"]) + 1
    
    # 正反双方角色设定
    roles = {
        "pro": "安全风险分析师（正方）：认为内容存在安全风险",
        "con": "内容宽容度分析师（反方）：认为内容安全可接受"
    }
    
    round_records = []
    for role, desc in roles.items():
        # 添加历史辩论记录
        history = "\n".join(state["debate_rounds"][-2:]) if state["debate_rounds"] else "无"
        
        # 获取辩论观点
        response = debater_model.invoke(
            f"{context}\n"
            f"## 你现在的角色\n{desc}\n"
            f"## 历史辩论记录\n{history}\n\n"
            f"请提出第{round_num}轮辩论观点（200字以内）："
        )
        round_records.append(f"{role.upper()}_ROUND{round_num}: {response.content}")
    
    # 更新辩论记录
    new_debate_rounds = state["debate_rounds"] + ["\n".join(round_records)]
    return {"debate_rounds": new_debate_rounds}

def summarize_node(state: AgentState) -> Dict[str, Any]:
    """总结节点：生成最终判断"""
    # 整理辩论记录
    debate_history = "\n\n".join([
        f"## 第{i+1}轮辩论\n{round}" 
        for i, round in enumerate(state["debate_rounds"])
    ])
    
    # 生成最终总结
    response = planner_model.invoke(
        f"### 任务说明\n{state['instruction']}\n\n"
        f"### 原始文本\n{state['text']}\n\n"
        f"### 图像描述\n{state['image_description']}\n\n"
        f"### 背景知识\n{state['search_results'][:1000]}\n\n"
        f"### 辩论记录\n{debate_history}\n\n"
        "请根据以上信息完成：\n"
        "1. 内容说明：概括输入的多模态内容\n"
        "2. 安全判断：'安全'或'存在风险'\n"
        "3. 详细原因：200字左右的分析\n"
        "按以下格式回复：\n"
        "【内容说明】...\n【安全判断】...\n【判断原因】..."
    )
    
    return {"final_summary": response.content}

# 4. 构建LangGraph
# 定义状态图
graph = StateGraph(AgentState)

# 添加节点
graph.add_node("preprocess", preprocess_node)
graph.add_node("plan", plan_node)
graph.add_node("search", search_node)
graph.add_node("debate", debate_node)
graph.add_node("summarize", summarize_node)

# 设置入口点
graph.set_entry_point("preprocess")

# 添加边连接
graph.add_edge("preprocess", "plan")
graph.add_edge("plan", "search")
graph.add_edge("search", "debate")

# 设置辩论循环（三轮）
graph.add_conditional_edges(
    "debate",
    # 条件判断函数
    lambda state: "summarize" if len(state["debate_rounds"]) >= 3 else "debate",
    {  # 路由选项
        "debate": "debate",
        "summarize": "summarize"
    }
)

graph.add_edge("summarize", END)

# 编译图
app = graph.compile()


# 5. 辅助函数
def encode_image(image_path: str) -> str:
    """将图像编码为Base64字符串"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# 6. 执行示例
if __name__ == "__main__":
    # 准备输入数据
    input_data = {
        "instruction": "判断以下内容是否包含暴力或危险元素",
        "text": "",
        "image": encode_image("../data/example.jpg")  
    }
    
    # 运行智能体
    final_state = app.invoke(input_data)
    
    # 打印最终结果
    print("\n" + "="*50)
    print("最终安全判断结果:")
    print("="*50)
    print(final_state["final_summary"])
    print("="*50)
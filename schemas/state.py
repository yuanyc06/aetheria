# schemas/state.py
from typing import TypedDict, Annotated, Sequence, List, Dict, Any
import operator
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    """工作流状态定义"""
    # 指令
    instruction: str

    # 原始输入
    raw_input: Dict[str, Any]
    
    # 识别出的模态类型
    modalities: List[str]
    
    # 转换后的文本描述
    translated_text: str
    
    # 支持者收集的背景信息
    background: str
    
    # 辩论历史
    debate_history: Annotated[Sequence[BaseMessage], operator.add]
    
    # 最终裁决和报告
    verdict: Dict[str, str]
    
    # 处理状态
    status: str
# graph/workflow.py
from langgraph.graph import StateGraph, END
from schemas.state import AgentState
from agents.preprocessor import PreprocessorAgent
from agents.planner import PlannerAgent
from agents.supporter import SupporterAgent
from agents.debaters import DebaterAgent
from agents.arbitrator import ArbitratorAgent
from utils.logger import get_logger, log_state_transition

logger = get_logger(__name__)

def create_workflow():
    logger.info("开始创建工作流")
    
    # 实例化智能体
    preprocessor = PreprocessorAgent()
    planner = PlannerAgent()
    supporter = SupporterAgent()
    debater = DebaterAgent()
    arbitrator = ArbitratorAgent()
    
    logger.info("所有智能体已实例化")
    
    # 定义工作流
    workflow = StateGraph(AgentState)
    
    # 添加节点
    workflow.add_node("preprocess", lambda state: preprocessor.process(state))
    workflow.add_node("plan", lambda state: planner.plan(state))
    workflow.add_node("supporter", lambda state: supporter.collect_background(state))
    workflow.add_node("debate", lambda state: debater.debate(state))
    workflow.add_node("arbitrator", lambda state: arbitrator.make_verdict(state))
    
    # 设置入口点
    workflow.set_entry_point("preprocess")
    
    # 添加边
    workflow.add_edge("preprocess", "plan")
    workflow.add_edge("plan", "supporter")
    workflow.add_edge("supporter", "debate")
    workflow.add_edge("debate", "arbitrator")
    workflow.add_edge("arbitrator", END)
    
    # 添加自定义状态转换日志
    def log_transition(from_node, to_node, state):
        log_state_transition(logger, from_node, to_node, state)
        return to_node
    
    workflow.add_conditional_edges(
        "plan",
        lambda state: state.get("next", "supporter"),
        log_transition
    )
    
    # 编译工作流
    compiled_workflow = workflow.compile()
    logger.info("工作流编译完成")
    
    return compiled_workflow

# 全局工作流实例
safety_workflow = create_workflow()
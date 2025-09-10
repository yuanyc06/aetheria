# agents/planner.py
from schemas.state import AgentState
from utils.logger import get_logger, log_execution

logger = get_logger(__name__)

class PlannerAgent:
    def __init__(self):
        from config import settings
        self.llm = settings.get_llm("planner")
        logger.info("规划者智能体已初始化，使用模型: %s", self.llm.model_name)

    @log_execution()
    def plan(self, state: AgentState) -> dict:
        """规划节点 - 定义工作流程"""
        logger.info("开始规划工作流程")
        
        # 实际应用中，这里可以根据内容复杂度添加更复杂的规划逻辑
        next_step = "supporter"
        
        logger.info("规划完成，下一步: %s", next_step)
        
        return {
            "next": next_step,
            "status": "planned"
        }
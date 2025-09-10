# agents/arbitrator.py
from langchain_core.messages import HumanMessage
from schemas.state import AgentState
from tools.tool_pool import tool_pool
from utils.logger import get_logger, log_execution
from config import settings

logger = get_logger(__name__)

class ArbitratorAgent:
    def __init__(self):        
        self.llm = settings.get_llm("arbitrator")
        logger.info("仲裁者智能体已初始化，使用模型: %s", self.llm.model_name)

    
    @log_execution()
    def make_verdict(self, state: AgentState) -> dict:
        """Arbitration Node - Make the Final Verdict"""
        logger.info("Starting final verdict generation")
        
        # Prepare debate history summary
        debate_summary = "\n".join([f"{msg.content}" for msg in state["debate_history"]])

        # Generate risk assessment report
        report_prompt = [
            HumanMessage(content=settings.PROMPT_TEMPLATES["arbitrator_prompt"].format(
                            instruction=state['instruction'],
                            raw_input=state['raw_input'],
                            translated_text=state['translated_text'],
                            background=state['background'],
                            debate_summary=debate_summary
                        ))
                                        ]
        
        logger.debug("Report prompt length: %d", len(report_prompt[0].content))
        
        # Get the final report
        response = self.llm.invoke(report_prompt)
        report_content = response.content
        
        # Check if tool verification is needed
        if "Tool Verification:" in report_content:
            tool_result = self._verify_with_tools(report_content, state)
            report_content += f"\n\nTool Verification Result:\n{tool_result}"
        
        # Extract risk decision
        # risk_decision = "Yes" if "Risk exists:Yes" in report_content else "No"
        if "## Risk Exists:" in report_content:
            # Find the line containing the risk assessment
            lines = report_content.split('\n')
            for line in lines:
                if line.startswith("## Risk Exists:"):
                    value = line.replace("## Risk Exists:", "").strip()
                    risk_decision = "Yes" if value == "Yes" else "No"
                    break
                else:
                    risk_decision = "No" # Default value
        
        logger.info("Risk assessment completed, decision: %s", risk_decision)
        logger.debug("Report summary: %s", report_content[:200] + "...")
        
        return {
            "verdict": {
                "decision": risk_decision,
                "report": report_content,  
                "risk_decision": risk_decision,
            },
            "status": "completed"
        }


    def _get_tools_description(self) -> str:
        """获取可用工具的描述"""
        tools = tool_pool.list_tools()
        return "\n".join([f"- {name}: {desc}" for name, desc in tools.items()])
    
    def _verify_with_tools(self, report: str, state: AgentState) -> str:
        """使用工具验证报告结论"""
        try:
            # 提取需要验证的部分
            if "工具验证:" in report:
                verification_section = report.split("工具验证:")[1].split("\n\n")[0].strip()
                
                # 使用文本分析工具验证风险
                if "文本" in verification_section and "text" in state["raw_input"]:
                    text_content = state["raw_input"]["text"]
                    return tool_pool.execute_tool(
                        "text_safety_check", 
                        text_content,
                        f"验证以下结论: {verification_section}"
                    )
                
                return "未识别到可验证的具体内容"
            return "未识别到验证请求"
        except Exception as e:
            logger.error("工具验证失败: %s", str(e))
            return f"验证错误: {str(e)}"
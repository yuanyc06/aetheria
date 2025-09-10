# agents/debaters.py
from dataclasses import dataclass
from langchain_core.messages import HumanMessage
from schemas.state import AgentState
from tools.tool_pool import tool_pool
from utils.logger import get_logger, log_execution
from config import settings
logger = get_logger(__name__)


@dataclass
class DebateTurn:
    role: str  # "strict" or "lenient"
    content: str
class DebaterAgent:
    def __init__(self):       
        self.llm = settings.get_llm("debaters")
        self.aligner_llm = settings.get_llm("aligner")  # 对齐者模型
        logger.info("辩论者智能体已初始化，使用模型: %s", self.llm.model_name)
        logger.info("对齐者智能体已初始化，使用模型: %s", self.aligner_llm.model_name)
        self.tool_pool = tool_pool

        # 最大允许对齐者要求辩论者重做的次数（每一轮、每个角色）
        self.MAX_CORRECTIONS = 1

    @log_execution()
    def debate(self, state: AgentState) -> dict:
        """辩论节点 - 两个角色 + 对齐者多轮辩论
        - 辩论者（debaters）**不再调用任何工具**，输出只专注于论点。
        - 对齐者（aligner）负责检查辩论者提及的**非文本模态**信息，决定是否调用工具验证。
        - 若对齐者经验证后认为辩论者描述不正确，会给出纠正意见，触发该辩论者在本轮重做（最多 self.MAX_CORRECTIONS 次）。
        """
        from config import settings

        logger.info("开始辩论环节，计划轮数: %d", settings.DEBATE_ROUNDS)

        # 定义辩论角色 - 严格标准和宽松标准
        debaters = settings.PROMPT_TEMPLATES["debaters_role"]

        # 对齐者角色
        aligner_role = {
            "name": "对齐者",
            "description": "你是一个非文本模态内容检查专家，检查用户是否提到的非文本模态信息是否准确，现在已经有了一些非文本模态信息，必要时可以调用工具验证，并提出纠正建议"

        }

        # 确保辩论历史存在
        if "debate_history" not in state:
            state["debate_history"] = []
        if "end" not in state:
            state["end"] = False

        # 进行多轮辩论
        for round_num in range(1, settings.DEBATE_ROUNDS + 1):
            logger.info("开始第 %d/%d 轮辩论", round_num, settings.DEBATE_ROUNDS)
            if state["end"]:
                break

            for debater in debaters:
                logger.debug("准备 %s 的辩论观点", debater["name"])
                if state["end"]:
                    break
                
                # 获取对方上一次的发言（如果有）
                opponent_turn = None
                self_turn = None

                # if state["debate_history"]:
                #     # 查找对方上一次发言
                #     for msg in reversed(state["debate_history"]):
                #         if isinstance(msg, HumanMessage) and debater["name"] not in msg.content:
                #             # 提取对方发言内容
                #             content_parts = msg.content.split(":", 1)
                #             if len(content_parts) > 1:
                #                 opponent_turn = DebateTurn(
                #                     role="strict" if "严格" in content_parts[0] else "lenient",
                #                     content=content_parts[1].strip()
                #                 )
                #             break
                
                for msg in reversed(state["debate_history"]):
                    if isinstance(msg, HumanMessage):
                        content_parts = msg.content.split(":", 1)
                        if len(content_parts) > 1:
                            role = "strict" if "strict" in content_parts[0].lower() else "lenient"
                            content = content_parts[1].strip()
                            turn = DebateTurn(role=role, content=content)
                            # 判断发言方：根据 debater["name"] 是否在前缀里区分
                            if debater["name"] not in content_parts[0]:
                                if opponent_turn is None:
                                    opponent_turn = turn
                            else:
                                if self_turn is None:
                                    self_turn = turn
                        # 如果两个都已找到，就可以提前终止遍历
                        if opponent_turn and self_turn:
                            break

                # 每个角色可能需要被对齐者要求重做多次
                attempt = 0
                while attempt < self.MAX_CORRECTIONS:
                    if state["end"]:
                        break

                    attempt += 1
                    logger.debug("%s 第%d轮 第%d次尝试", debater["name"], round_num, attempt)

                    # 构建辩论提示
                    prompt = self._build_debater_prompt(
                        debater, state, round_num, opponent_turn
                    )
                    
                    # 获取辩论观点（辩论者不能也不应调用工具）
                    response = self.llm.invoke(prompt)
                    debate_content = response.content


                    # # 如果辩论者文本中包含工具调用指令，直接忽略以防误用
                    # if "使用工具:" in debate_content:
                    #     logger.warning("检测到辩论者尝试调用工具，已忽略该指令。角色: %s", debater["name"])
                    #     debate_content = '\n'.join(
                    #         [line for line in debate_content.split('\n') if '使用工具:' not in line]
                    #     )

                    # 将对齐者之前的反馈（如果有）从历史中移除，以便添加更新后的版本
                    if attempt > 1 and state["debate_history"] and "对齐者" in state["debate_history"][-1].content:
                        state["debate_history"].pop()
                    
                    # 添加辩论者观点到历史
                    new_message = HumanMessage(
                        content=f"===={debater['name']} Round {round_num} viewpoint:====\n {debate_content}\n"
                    )

                    state["debate_history"].append(new_message)
                    
                    if "I agree with your viewpoint" in debate_content:
                        state["end"] = True
                        break

                    # 对齐者检查（对齐者可以选择是否调用工具检查非文本模态内容）
                    # aligner_feedback = self._get_aligner_feedback(
                    #     aligner_role, debater, debate_content, state
                    # )
                    
                    # 将对齐者反馈写入历史
                    # state["debate_history"].append(HumanMessage(
                    #     content=f"{aligner_role['name']} 第{round_num}轮反馈: {aligner_feedback}"
                    # ))

                    # 判断对齐者是否认为需要纠正
                    # if self._is_correction_needed(aligner_feedback):
                    #     logger.info("对齐者认为描述有误，需要纠正，角色 %s 在第%d轮第%d次将重做。", debater["name"], round_num, attempt)
                        
                    #     # 若超过最大重试次数，则记录并跳出重试循环，保留对齐者反馈
                    #     if attempt >= self.MAX_CORRECTIONS:
                    #         logger.warning("%s 在第%d轮达到最大重试次数，不再重做。", debater["name"], round_num)
                    #         break
                            
                    #     # 将对齐者的纠正意见加入历史，并提示辩论者基于纠正意见重做
                    #     correction_note = HumanMessage(
                    #         content=(
                    #             f"请基于对齐者反馈修改并重新给出第{round_num}轮观点。对齐者反馈:\n{aligner_feedback}"
                    #         )
                    #     )
                    #     state["debate_history"].append(correction_note)
                        
                    #     # 重做：回到循环的顶部（debater 重新生成观点）
                    #     continue
                    
                    # # 若不需要纠正或不再重试，则完成该角色本轮辩论
                    # logger.debug("%s 第%d轮确认完成（attempt=%d）", debater["name"], round_num, attempt)
                    # break

        logger.info("辩论完成，总辩论记录数: %d", len(state.get("debate_history", [])))

        return {
            "status": "debate_completed"
        }
    
    def _build_debater_prompt(self, debater, state, round_num, opponent_turn):
        """构建辩论者提示"""
        # 基础提示
        prompt_parts = [
            f"Content Safety Risk Assessment Debate - Round {round_num}\n",
            f"User Instruction:\n{state['instruction']}\n",
            f"User Input:\n{state['translated_text']}\n",
        ]
        
        # 添加背景信息（如果有）
        if state.get('background'):
            # prompt_parts.append(f"背景信息:\n{state['background']}\n（背景信息并非用户输入，仅供参考）\n")
            prompt_parts.append(
                f"Background Information:\n{state['background']}\n(This background information is not part of the user input and is for reference only)\n"
            )

        
        # Add debater role and stance
        prompt_parts.append(f"Your role: {debater['name']} ({debater['stance']})\n\n")

        # Add debate history
        if state["debate_history"]:
            prompt_parts.append("Debate History:\n")
            for msg in state["debate_history"][-6:]:  # Keep only the most recent few messages
                if opponent_turn:
                    opponent_name = "Strict Content Safety Reviewer" if opponent_turn.role == "strict" else "Content Safety Reviewer"
                
                content = msg.content
                # content = content.replace(opponent_name, "Human")
                
                prompt_parts.append(f"{content}\n")

        
        # 添加对手上一轮观点（如果有）
        # if opponent_turn:
        #     opponent_name = "严格标准审核员" if opponent_turn.role == "strict" else "宽松标准审核员"
        #     # prompt_parts.append(f"\n对方上一轮观点: {opponent_name}: {opponent_turn.content}\n")
        #     prompt_parts.append(f"\n人类上一轮观点:  {opponent_turn.content}\n")
        
        # 最终指令
        prompt_parts.append(settings.PROMPT_TEMPLATES["debate_next"])
        
        return [HumanMessage(content="".join(prompt_parts))]
    
    def _get_aligner_feedback(self, aligner_role, debater, debate_content, state):
        """获取对齐者反馈"""
        # 构建对齐者提示
        aligner_prompt = [
            HumanMessage(content=(
                f"{aligner_role['description']}：请检查以下辩论观点是否涉及非文本模态（例如图片、视频、音频等）信息，"
                "并判断这些模态信息是否被描述正确。若需要使用工具验证，请以明确的格式回复包含 `使用工具:工具名:参数` 的行，然后在工具返回后给出最终结论与纠正建议。\n"
                f"辩论者: {debater['name']}\n"
                f"辩论者观点:\n{debate_content}\n"
            ))
        ] + state["debate_history"][-5:]  # 只包含最近的历史
        
        aligner_response = self.aligner_llm.invoke(aligner_prompt)
        aligner_feedback = aligner_response.content

        # 如果对齐者请求调用工具，则执行并把结果补回给对齐者，再让对齐者给出最后结论
        if "使用工具:" in aligner_feedback:
            tool_result = self._handle_tool_request(aligner_feedback)
            logger.info("对齐者请求工具并返回结果: %s", tool_result)

            # 把工具结果作为上下文让对齐者再次给出最终反馈
            followup_prompt = [
                HumanMessage(content=(
                    f"这是工具返回的结果:\n{tool_result}\n\n请基于该工具结果，给出最终结论（是否存在模态描述错误）和具体的纠正建议。"
                ))
            ] + state["debate_history"][-3:]  # 只包含最近的历史
            
            second_align_resp = self.aligner_llm.invoke(followup_prompt)
            aligner_feedback = second_align_resp.content
            
        return aligner_feedback
    
    def _is_correction_needed(self, aligner_feedback):
        """判断是否需要纠正"""
        # 简单关键词判断对齐者是否认为需要纠正
        negative_indicators = ["错误", "不准确", "纠正", "重做", "不认同", "不同意"]
        return any(indicator in aligner_feedback for indicator in negative_indicators)
    
    def _handle_tool_request(self, aligner_feedback):
        """处理对齐者的工具请求"""
        # 解析工具请求
        lines = aligner_feedback.split('\n')
        tool_line = None
        for line in lines:
            if line.startswith('使用工具:'):
                tool_line = line
                break
        
        if not tool_line:
            return "未找到有效的工具调用指令"
        
        # 解析工具名和参数
        parts = tool_line.split(':', 2)
        if len(parts) < 3:
            return "工具调用格式错误，应为: 使用工具:工具名:参数"
        
        tool_name = parts[1].strip()
        tool_params = parts[2].strip()
        
        # 调用工具
        try:
            tool = self.tool_pool.get_tool(tool_name)
            if tool:
                result = tool.execute(tool_params)
                return f"工具 {tool_name} 执行结果: {result}"
            else:
                return f"未找到工具: {tool_name}"
        except Exception as e:
            return f"工具执行出错: {str(e)}"



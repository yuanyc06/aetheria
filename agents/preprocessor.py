# agents/preprocessor.py
import logging
from schemas.state import AgentState
from tools.tool_pool import tool_pool
from utils.logger import get_logger, log_execution
from config import settings

logger = get_logger(__name__)

class PreprocessorAgent:
    def __init__(self):
        self.llm = settings.get_llm("preprocessor")
        logger.info("预处理智能体已初始化，使用模型: %s", self.llm.model_name)

    @log_execution()
    def process(self, state: AgentState) -> dict:
        """预处理节点 - 识别模态并转换内容"""
        input_data = state["raw_input"]
        logger.info("开始预处理输入数据")
        
        modalities = []
        translated_text = ""
        
        # 记录输入模态
        logger.debug("检测输入模态: %s", list(input_data.keys()))
        
        # 识别文本内容
        if input_data["text"] and input_data["text"].strip():
            modalities.append("text")
            text_content = input_data["text"]
            
            # 使用self.llm处理文本
            prompt = "请分析这段文本内容，并识别可能存在的安全风险"
            messages = [
                {"role": "user", "content": prompt + "\n\n" + text_content}
            ]
            
            translated_text += f"- User Input Text: {text_content}\n"
            
            logger.debug("已处理文本内容 (长度: %d)", len(text_content))
        
        # 识别并处理图像
        if input_data["image"] and input_data["image"].strip():
            modalities.append("image")
            logger.debug("开始处理图像数据 (长度: %d)", len(input_data["image"]))
            
            # 使用self.llm处理图像
            from multimodal.vision import VisionProcessor
            
            # 准备提示词和图像数据
            prompt = settings.PROMPT_TEMPLATES["preprocessor_image_prompt"].format(
                instruction=state["instruction"],
                input_text=state["raw_input"]["text"] or "No text entered by the user."
            )
            messages = [
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{input_data['image']}", "detail": "high"}}
                ]}
            ]
            
            # 使用self.llm进行多模态处理
            response = self.llm.invoke(messages)
            img_desc = response.content
            translated_text += f"Image description: {img_desc}\n"
            
            logger.info("图像处理完成: %s", img_desc[:50] + "...")
        
        # 识别并处理音频
        if input_data["audio"] and input_data["audio"].strip():
            modalities.append("audio")
            logger.debug("开始处理音频数据 (长度: %d)", len(input_data["audio"]))
            
            # 使用self.llm处理音频
            # 准备提示词和音频数据
            prompt = "请转录这段音频内容，并识别可能存在的安全风险"
            messages = [
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "audio", "audio": {"url": f"data:audio/mp3;base64,{input_data['audio']}"}}
                ]}
            ]
            
            # 使用self.llm进行多模态处理
            response = self.llm.invoke(messages)
            audio_desc = response.content
            translated_text += f"- Audio transcription: {audio_desc}\n"
            
            logger.info("音频处理完成: %s", audio_desc[:50] + "...")
        
        # 识别并处理视频
        if input_data["video"] and input_data["video"].strip():
            modalities.append("video")
            logger.debug("开始处理视频数据 (长度: %d)", len(input_data["video"]))
            
            # 使用self.llm处理视频
            # 准备提示词和视频数据
            prompt = "请分析这段视频内容，并识别可能存在的安全风险"
            messages = [
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "video", "video": {"url": f"data:video/mp4;base64,{input_data['video']}"}}
                ]}
            ]
            
            # 使用self.llm进行多模态处理
            response = self.llm.invoke(messages)
            video_desc = response.content
            translated_text += f"- Video analysis: {video_desc}\n"
            
            logger.info("视频处理完成: %s", video_desc[:50] + "...")
        
        # 如果没有识别到任何模态
        if not modalities:
            translated_text = "未识别到有效输入内容"
            logger.warning("未识别到任何有效输入模态")
        
        logger.info("预处理完成，识别模态: %s", modalities)
        
        return {
            "modalities": modalities,
            "translated_text": translated_text,
            "status": "preprocessed"
        }
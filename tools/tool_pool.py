# tools/tool_pool.py
from langchain.tools import BaseTool
from typing import Dict, Type, Any, Optional
from multimodal.vision import VisionProcessor
from multimodal.audio import AudioProcessor
from multimodal.video import VideoProcessor
from utils.logger import get_logger
import inspect

logger = get_logger(__name__)

class ToolPool:
    """多模态工具池，用于管理和调用各种处理工具"""
    
    def __init__(self):
        self.tools: Dict[str, BaseTool] = {}
        self._register_default_tools()
    
    def _register_default_tools(self):
        """注册默认的多模态处理工具"""
        # 视觉工具
        self.register_tool(
            name="image_analyzer",
            description="分析图像内容并生成详细描述",
            func=VisionProcessor.image_to_text,
            args_schema=inspect.signature(VisionProcessor.image_to_text)
        )
        
        # 音频工具
        self.register_tool(
            name="audio_transcriber",
            description="将音频内容转换为文字稿",
            func=AudioProcessor.audio_to_text,
            args_schema=inspect.signature(AudioProcessor.audio_to_text)
        )
        
        # 视频工具
        self.register_tool(
            name="video_analyzer",
            description="分析视频内容并生成详细描述",
            func=VideoProcessor.video_to_text,
            args_schema=inspect.signature(VideoProcessor.video_to_text)
        )
        
        # 文本分析工具
        self.register_tool(
            name="text_safety_checker",
            description="检查文本内容是否存在安全风险",
            func=self.text_safety_check,
            args_schema=inspect.signature(self.text_safety_check)
        )
        
        logger.info("已注册默认工具: %s", list(self.tools.keys()))
    
    def register_tool(self, name: str, description: str, func: callable, args_schema: inspect.Signature):
        """注册新工具"""
        class CustomTool(BaseTool):
            def __init__(self):
                super().__init__(name=name, description=description)
                self._func = func

            def _run(self, *args, **kwargs):
                return self._func(*args, **kwargs)

            async def _arun(self, *args, **kwargs):
                import asyncio
                return await asyncio.to_thread(self._func, *args, **kwargs)
        
        self.tools[name] = CustomTool()
        logger.debug("已注册工具: %s", name)
    
    def get_tool(self, name: str) -> Optional[BaseTool]:
        """获取指定工具"""
        return self.tools.get(name)
    
    def list_tools(self) -> Dict[str, str]:
        """列出所有可用工具及其描述"""
        return {name: tool.description for name, tool in self.tools.items()}
    
    def execute_tool(self, tool_name: str, *args, **kwargs) -> Any:
        """执行指定工具"""
        tool = self.get_tool(tool_name)
        if not tool:
            logger.error("未找到工具: %s", tool_name)
            return f"错误: 未找到工具 '{tool_name}'"
        
        logger.info("执行工具: %s", tool_name)
        try:
            result = tool.run(*args, **kwargs)
            logger.debug("工具执行结果: %s", result[:100] + "..." if isinstance(result, str) else result)
            return result
        except Exception as e:
            logger.exception("工具执行失败: %s", tool_name)
            return f"工具执行错误: {str(e)}"
    
    @staticmethod
    def text_safety_check(text: str, context: str = "") -> str:
        """检查文本内容是否存在安全风险"""
        from config import settings
        
        llm = settings.get_llm("tool_text_safety")
        prompt = (
            f"分析以下文本是否存在安全风险:\n"
            f"文本内容: {text}\n"
            f"上下文: {context}\n\n"
            "请按以下格式回答:\n"
            "风险评估: [安全/低风险/中风险/高风险]\n"
            "风险类型: [如暴力、仇恨言论等]\n"
            "详细解释: [解释原因]"
        )
        
        response = llm.invoke(prompt)
        return response.content

# 全局工具池实例
tool_pool = ToolPool()
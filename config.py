# config.py
import os
import logging
from logging.config import dictConfig
from prompts import preprocessor_prompt,supporter_prompt,debaters_prompt,arbitrator_prompt
from dotenv import load_dotenv

load_dotenv()  



class Settings:
    # API设置
    OPENKEY_API_KEY = os.getenv("OPENKEY_API_KEY")
    OPENKEY_BASE_URL = os.getenv("OPENKEY_BASE_URL")
    AZURE_API_KEY = os.getenv("AZURE_API_KEY")
    AZURE_BASE_URL = os.getenv("AZURE_BASE_URL")
    AZURE_API_VERSION = os.getenv("AZURE_API_VERSION")
    AZURE_MODELS = ['gpt-35-turbo', 'gpt-4.1','gpt-4.1-mini','gpt-4o', 'gpt-4o-mini', 'o1', 'o3-mini']

    USE_AZURE = True

    
    # 默认模型设置
    DEFAULT_MODEL = "gpt-4o"

    # 多模态工具配置
    MULTIMODAL_TOOLS = {
        "image": "gpt-4.1",
        "audio": "doubao-1.5-audio-pro-250328",
        "video": "doubao-1.5-video-pro-250328"
    }
    
    # 智能体模型配置
    AGENT_MODELS = {
        "preprocessor": "gpt-4o",
        "planner": "gpt-4o-mini",
        "supporter": "gpt-4o-mini",
        "debaters": "gpt-4o-mini",
        "arbitrator": "gpt-4o-mini",
        "aligner": "gpt-4o",
        "tool_text_safety": "gpt-35-turbo"  # 新增工具专用模型
    }

    # 提示模板
    PROMPT_TEMPLATES = {
        "image_to_text": preprocessor_prompt.Image_to_text_prompt,
        "collect_background": supporter_prompt.collect_background_en,
        "summarize_background": supporter_prompt.summarize_background_en,
        "debaters_role": debaters_prompt.debaters_role_en,
        "debate_next": debaters_prompt.debate_next_en,
        "arbitrator_prompt": arbitrator_prompt.arbitrator_en,
        "preprocessor_image_prompt": preprocessor_prompt.preprocessor_image_prompt_en,



        "preprocessor": "分析用户输入，提取关键信息",
        "planner": "根据用户需求和背景，规划处理流程",
        "supporter": "支持用户需求，提供相关信息",
        "debaters": "参与辩论，表达观点和argument",
        "arbitrator": "判断辩论结果，确定最终输出",
        "aligner": "调整输出，确保符合用户需求"
    }

    

    # 辩论设置
    DEBATE_ROUNDS = 2
    
    # 多模态处理设置
    MAX_IMAGE_SIZE = (256, 256)
    MAX_AUDIO_DURATION = 60  # 秒
    
    # 日志设置
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s"
    LOG_FILE = "safety_assessment.log"
    
    @classmethod
    def get_llm(cls, agent_name: str = None):
        """获取特定智能体的语言模型"""
        from langchain_openai import ChatOpenAI, AzureChatOpenAI,AzureOpenAI

        
        model = cls.AGENT_MODELS.get(agent_name, cls.DEFAULT_MODEL)

        if model in cls.AZURE_MODELS:
            return AzureChatOpenAI(
                api_version=cls.AZURE_API_VERSION,
                azure_endpoint=cls.AZURE_BASE_URL,
                api_key=cls.AZURE_API_KEY,
                model=model,
                temperature=0.1  # 降低随机性
            )

        else:
            return ChatOpenAI(
                api_key=cls.OPENKEY_API_KEY,
                base_url=cls.OPENKEY_BASE_URL,
                model=model,
                temperature=0.1  # 降低随机性
            )
    
    @classmethod
    def set_agent_model(cls, agent_name: str, model_name: str):
        """为特定智能体设置模型"""
        if agent_name in cls.AGENT_MODELS:
            cls.AGENT_MODELS[agent_name] = model_name
        else:
            raise ValueError(f"未知的智能体名称: {agent_name}")

    @classmethod
    def configure_logging(cls):
        """配置日志系统"""
        logging_config = {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "standard": {
                    "format": cls.LOG_FORMAT,
                    "datefmt": "%Y-%m-%d %H:%M:%S"
                }
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": "standard",
                    "level": cls.LOG_LEVEL
                },
                "file": {
                    "class": "logging.FileHandler",
                    "filename": cls.LOG_FILE,
                    "formatter": "standard",
                    "level": "DEBUG"
                }
            },
            "loggers": {
                "": {  # 根记录器
                    "handlers": ["console", "file"],
                    "level": "DEBUG",
                    "propagate": False
                },
                "agents": {
                    "level": "INFO",
                    "propagate": False
                },
                "multimodal": {
                    "level": "INFO",
                    "propagate": False
                },
                "graph": {
                    "level": "INFO",
                    "propagate": False
                },
                "tools": {
                    "level": "DEBUG",
                    "propagate": False
                }
            }
        }
        
        dictConfig(logging_config)
        logger = logging.getLogger(__name__)
        logger.info("日志系统已配置，日志级别: %s", cls.LOG_LEVEL)


settings = Settings()
settings.configure_logging()

if __name__ == "__main__":
    settings = Settings()
    settings.AGENT_MODELS["supporter"] = "gpt-4o-mini"
    llm = settings.get_llm("supporter")
    # llm.invoke("你好")
    print(llm.invoke("你好"))


import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import AzureOpenAIEmbeddings, OpenAIEmbeddings
from langchain.tools.retriever import create_retriever_tool
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from pydantic import BaseModel, Field
from typing import Literal
from utils.logger import get_logger
from config import settings

logger = get_logger(__name__)


class GradeDocuments(BaseModel):
    """Grade documents using a binary score for relevance check."""

    binary_score: str = Field(
        description="Relevance score: 'yes' if relevant, or 'no' if not relevant"
    )


class RAGTool:
    def __init__(self):
        """初始化 RAG 工具"""
        self.response_model = self._get_llm()
        self.grader_model = self._get_llm()
        self.vectorstore = None
        self.retriever = None
        self.retriever_tool = None
        self._initialize_rag()


    def _get_llm(self):
        """获取语言模型，使用与 config 相同的配置方式"""
        if settings.USE_AZURE:
            return AzureChatOpenAI(
                api_key=settings.AZURE_API_KEY,
                azure_endpoint=settings.AZURE_BASE_URL,
                azure_deployment="gpt-4o",
                api_version=settings.AZURE_API_VERSION,
                temperature=0,
            )
        else:
            return ChatOpenAI(
                api_key=settings.OPENKEY_API_KEY,
                base_url=settings.OPENKEY_BASE_URL,
                model="gpt-4-turbo",
                temperature=0,
            )

    def _get_embeddings(self):
        """获取嵌入模型，使用与 config 相同的配置方式"""
        if settings.USE_AZURE:
            return AzureOpenAIEmbeddings(
                api_key=settings.AZURE_API_KEY,
                azure_endpoint=settings.AZURE_BASE_URL,
                azure_deployment="text-embedding-ada-002",
                api_version=settings.AZURE_API_VERSION,
            )
        else:
            return OpenAIEmbeddings(
                api_key=settings.OPENKEY_API_KEY,
                base_url=settings.OPENKEY_BASE_URL,
                model="text-embedding-ada-002",
            )

    def _initialize_rag(self):
        """初始化 RAG 系统"""
        try:
            reports_dir = "reports"
            if not os.path.exists(reports_dir):
                self.vectorstore = InMemoryVectorStore.from_documents(
                    [], embedding=self._get_embeddings()
                )
                self.retriever = self.vectorstore.as_retriever()
                return

            report_files = []
            for filename in os.listdir(reports_dir):
                if filename.endswith(".txt"):
                    filepath = os.path.join(reports_dir, filename)
                    report_files.append(filepath)

            if not report_files:
                logger.warning("reports 目录中没有找到历史报告文件")
                self.vectorstore = InMemoryVectorStore.from_documents(
                    [], embedding=self._get_embeddings()
                )
                self.retriever = self.vectorstore.as_retriever()
                return

            from langchain_core.documents import Document

            docs_list = []

            for filepath in report_files:
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        content = f.read()

                    doc = Document(
                        page_content=content,
                        metadata={
                            "source": filepath,
                            "filename": os.path.basename(filepath),
                        },
                    )
                    docs_list.append(doc)
                    logger.debug(f"加载历史报告: {os.path.basename(filepath)}")
                except Exception as e:
                    logger.error(f"读取报告文件失败 {filepath}: {e}")

            if not docs_list:
                logger.warning("没有成功加载任何历史报告")
                self.vectorstore = InMemoryVectorStore.from_documents(
                    [], embedding=self._get_embeddings()
                )
                self.retriever = self.vectorstore.as_retriever()
                return

            text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                chunk_size=200, chunk_overlap=50
            )
            doc_splits = text_splitter.split_documents(docs_list)

            self.vectorstore = InMemoryVectorStore.from_documents(
                documents=doc_splits, embedding=self._get_embeddings()
            )
            self.retriever = self.vectorstore.as_retriever()

            self.retriever_tool = create_retriever_tool(
                self.retriever,
                "retrieve_historical_cases",
                "Search and return information about historical safety assessment cases from local reports.",
            )

            logger.info(
                f"RAG 系统初始化成功，加载了 {len(docs_list)} 个历史报告文件，分割为 {len(doc_splits)} 个文档块"
            )
        except Exception as e:
            logger.error(f"RAG 系统初始化失败: {e}")
            self.vectorstore = None
            self.retriever = None
            self.retriever_tool = None

    def search_historical_cases(self, query: str, max_results: int = 5) -> str:
        """搜索历史案例"""
        if not self.retriever:
            return "RAG 系统未初始化，无法搜索历史案例"

        try:
            docs = self.retriever.get_relevant_documents(query, k=max_results)

            if not docs:
                return "未找到相关的历史案例"
            results = []
            for i, doc in enumerate(docs, 1):
                content = doc.page_content.strip()
                if len(content) > 200:
                    content = content[:200] + "..."
                results.append(f"案例 {i}:\n{content}\n")

            return "\n".join(results)
        except Exception as e:
            logger.error(f"搜索历史案例失败: {e}")
            return f"搜索历史案例时出错: {str(e)}"

    def grade_relevance(self, query: str, context: str) -> bool:
        """评估检索结果的相关性"""
        if not self.grader_model:
            return False

        try:
            GRADE_PROMPT = (
                "You are a grader assessing relevance of a retrieved document to a user question. \n "
                "Here is the retrieved document: \n\n {context} \n\n"
                "Here is the user question: {question} \n"
                "If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n"
                "Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."
            )

            prompt = GRADE_PROMPT.format(question=query, context=context)
            response = self.grader_model.with_structured_output(GradeDocuments).invoke(
                [{"role": "user", "content": prompt}]
            )

            return response.binary_score == "yes"
        except Exception as e:
            logger.error(f"评估相关性失败: {e}")
            return False


rag_tool = RAGTool()

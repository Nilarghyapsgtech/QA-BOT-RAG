from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough

from app.config import get_settings
from app.utils.logger import get_logger
from app.core.vector_store import VectorStoreService

logger=get_logger(__name__)
settings=get_settings()


RAG_PROMPT_TEMPLATE = """You are a helpful assistant. Answer the question based on the provided context.

If you cannot answer the question based on the context, say "I don't have enough information to answer that question."

Do not make up information. Only use the context provided.

Context:
{context}

Question: {question}

Answer:"""


def format_docs(docs:list[Document])->str:
    """Format documents into a single context string."""
    
    return "\n\n---\n\n".join(doc.page_content for doc in docs)

class RAGChain:
    def __init__(self,vector_store_service:VectorStoreService|None=None):
        self.vector_store=vector_store_service or VectorStoreService
        self.retriever=self.vector_store.get_retriever()
        self._evaluator=None

        self.llm=ChatOpenAI(
            model=settings.llm_model,
            temperature=settings.llm_temperature
        )
        self.prompt=ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
        self.chain=(
            {
                "context":self.retriever|format_docs,
                "question":RunnablePassthrough()
            }|self.prompt|self.llm|StrOutputParser()
        )
        logger.info(f"Ragchain initialized with llm={settings.llm_model} and retrievals at a time {settings.retrieval_k}")
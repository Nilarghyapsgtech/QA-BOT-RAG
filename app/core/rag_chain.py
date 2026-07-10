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

    @property
    def evaluator(self):
            if self._evaluator is None:
                from app.core.ragas_evaluation import RAGASEvaluator
                self._evaluator=RAGASEvaluator
            return self._evaluator
    
    def query(self,question:str)->str:
         try:
            answer=self.chain.invoke(question)
            logger.info("Query processed Successfully")
         except Exception as e:
              logger.error("Error in query processes with query {e}")
              raise

    def query_with_sources(self,question:str)->str:
         logger.info(f"Processing query with sources: {question[:100]}...")
         try:
              answer=self.chain.invoke(question)
              source_docs=self.retriever.invoke(question)

              sources=[
                   {
                    "content":(
                        doc.page_content[:500]+"..."
                        if len(doc.page_content)>500
                        else doc.page_content
                   ),
                   "metadata":doc.metadata} for doc in source_docs
              ]
              return {
                   "answer":answer,
                   "source_docs":sources
              }
         except Exception as e:
              logger.error("Error in query processes with query {e}")
              raise
         
    def aquery(self,question:str)->str:
         try:
            answer=self.chain.ainvoke(question)
            logger.info("Async Query processed Successfully")
         except Exception as e:
              logger.error("Error in async query processes with query {e}")
              raise
         
    def aquery_with_sources(self,question:str)->str:
         logger.info(f"Processing async query with sources: {question[:100]}...")
         try:
              answer=self.chain.ainvoke(question)
              source_docs=self.retriever.invoke(question)

              sources=[
                   {
                    "content":(
                        doc.page_content[:500]+"..."
                        if len(doc.page_content)>500
                        else doc.page_content
                   ),
                   "metadata":doc.metadata} for doc in source_docs
              ]
              return {
                   "answer":answer,
                   "source_docs":sources
              }
         except Exception as e:
              logger.error("Error in query processing with sources {e}")
              raise
         
    def stream(self,question:str):
         logger.info(f"Streaming query:{question[:500]}")
         try:
            for chunk in self.chain.stream(question):
              yield chunk
         except Exception as e:
              logger.error(f"Error in streaming response withj exception{e}")


    

    

              

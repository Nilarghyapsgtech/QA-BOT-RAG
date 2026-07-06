from functools import lru_cache
from app.config import get_settings
from langchain_openai import OpenAIEmbeddings
from app.utils.logger import get_logger

from dotenv import load_dotenv

load_dotenv()

logger=get_logger(__name__)

@lru_cache
def get_embeddings()->OpenAIEmbeddings:
    settings=get_settings()
    
    embeddings=OpenAIEmbeddings(
        model=settings.embedding_model,
    )
    logger.info(f"Initialized Embedding Model:{settings.embedding_model}")
    return embeddings

class EmbeddingService:
    def __init__(self):
        self.settings=get_settings()
        self.embeddings=get_embeddings()

    def embed_query(self,query:str)->list[float]:
        embeddings=self.embeddings.embed_query(query)
        logger.info(f"Embeddings generated for query {query[:50]} ....")
        return embeddings
    
    def embed_documents(self,documents:list[str])->list[list[float]]:
        embeddings=self.embeddings.embed_documents(documents)
        logger.info(f"Embeddings generated for {len(documents)} documents")
        return embeddings

    





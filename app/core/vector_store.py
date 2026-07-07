from functools import lru_cache
from app.config import get_settings
from app.utils.logger import get_logger

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance,VectorParams
from qdrant_client.http.exceptions import UnexpectedResponse

logger=get_logger()
settings=get_settings()

# Embedding Dimensions for text-3-embedding-small
EMBEDDING_DIMENSION=1536

@lru_cache
def get_qdrant_client()->QdrantClient:
    logger.info(f"Connecting to Qdrant at : {settings.qdrant_url}")
    client=QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key
    )
    logger.info(f"Qdrant connected Successfully")
    return client
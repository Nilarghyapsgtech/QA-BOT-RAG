from functools import lru_cache
from app.config import get_settings
from app.utils.logger import get_logger
from uuid import uuid4

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from app.core.embeddings import get_embeddings
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

class VectorStoreService:
    """Service for managing Vector Store Operations"""
    def __init__(self,collection_name:str|None=None):
        self.collection_name=settings.collection_name or collection_name
        self.client=get_qdrant_client()
        self.embeddings=get_embeddings()

        self.vector_store=QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=self.embeddings
        )

        logger.info(f"Vector Store Initialized for  collection:{self.collection_name}")

    
    def _ensure_collection(self)->None:
        """Ensure the collection exists, create if not."""
        try:
            collection_info=self.client.get_collection(self.collection_name)
            logger.info(f"Collection {self.collection_name} alreadfy exists with point {collection_info.points_count}")
            # Points refer to how many documents are available within that particular collection
        except UnexpectedResponse:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=EMBEDDING_DIMENSION,
                    distance=Distance.COSINE
                )
            )
            logger.info(f"Collection created with Collection Name:{self.collection_name}")

    def add_documents(self,documents:list[Document]):
        if not documents:
            logger.warning(f"No Documents Found")
            return []
        ids=[str(uuid4()) for _ in documents]
        self.vector_store.add_documents(documents,id=ids)
        logger.info(f"Successfully added {len(documents)} to collection")
        return ids
    
    def search(self,k:int,query:str)->list[Document]:

        k=k or settings.retrieval_k
        logger.debug(f"Searching Query: {[query[:50]]}..... (k={k})")

        results=self.vector_store.similarity_search(k=k)
        logger.debug(f"Found {len(results)} results")
        return results
    
    def search_with_scores(self,k:int,query:str)->list[tuple[Document,float]]:

        k=k or settings.retrieval_k
        logger.debug(f"Searching Query: {[query[:50]]}..... (k={k})")

        results=self.vector_store.similarity_search_with_score(k=k)
        logger.debug(f"Found {len(results)} results with scores")
        return results
    
    def get_retriever(self,k:int|None=None):
        k=k or settings.retrieval_k
        retriever=self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k":k}
        )
        return retriever
    
    def delete_collection(self,collection_name:str):
        try:
            self.client.delete_collection(collection_name)
            logger.info(f"Collection :{collection_name} Deleted")
        except UnexpectedResponse:
            logger.info(f"Collection :{collection_name} not found")

    def get_collection(self,collection_name:str)->dict:
        try:
            info=self.client.get_collection(collection_name)
            return {
                "name":self.collection_name,
                "points_count":info.points_count,
                "status":info.status.value,
                "indexed_vectors_count":info.indexed_vectors_count
            }
        except UnexpectedResponse:
            logger.info(f"Collection :{collection_name} not found")

    def health_check(self)->bool:
        try:
            self.client.get_collections()
            return True
        except Exception as e:
            logger.error(f"Vector Databse is Down due to error {e}")
            return False

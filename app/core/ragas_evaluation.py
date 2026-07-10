from typing import Any,Dict
import asyncio
from datasets import Dataset
from langchain_openai import OpenAIEmbeddings,ChatOpenAI
from ragas import evaluate
from ragas.metrics import _answer_relevance ,_faithfulness
from app.config import get_settings
from app.utils.logger import get_logger


logger=get_logger(__name__)
settings=get_settings()

class RAGASEvaluator:
    """Evaluator for Rag Response using Ragas Metrics"""
    def __init__(self):
        
        eval_llm_model=settings.ragas_embedding_model or settings.llm_model
        eval_llm_temperature=(
            settings.ragas_llm_temperature
            if settings.ragas_llm_temperature is not None
            else settings.llm_temperature
        )
        eval_embedding_model=settings.ragas_embedding_model or settings.embedding_model
        self.llm=ChatOpenAI(
            model=settings.llm_model,
            temperature=settings.llm_temperature
        )
        self.embedding_model=OpenAIEmbeddings(
            model=settings.embedding_model,

        )
        self.metrics=[_faithfulness,_answer_relevance]

        logger.info(
            f"RAGAS evaluator initialized - "
            f"LLM: {eval_llm_model} (temp={eval_llm_temperature}), "
            f"Embeddings: {eval_embedding_model}, "
            f"Metrics: {[metric.name for metric in self.metrics]}"
        )

    def _handle_evaluation_error(self,error:Exception)->dict[str,Any]:
        logger.error(f"Returning Fallback scores due to error {error}")
        return {
            "faithfullness":None,
            "answer_relevancy": None,
            "evaluation_time_ms": None,
            "error": str(error)
        }



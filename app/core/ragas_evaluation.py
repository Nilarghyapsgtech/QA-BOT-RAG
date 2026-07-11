from typing import Any,Dict
import asyncio
import time
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
    
    def prepare_dataset(self,question:str,answer:str,context:list[str])->Dataset:
        dataset={
            "question":[question],
            "answer":[answer],
            "context":[context],
        }
        logger.debug(
            f"Prepared dataset with {len(context)} contexts " f"for question: {question[:50]}..."
        )

        return Dataset.from_dict(dataset)
    
    def _evaluate_with_timeout(self,dataset:Dataset)->Dict[str,Any]:
        result=evaluate(
            dataset=dataset,
            llm=self.llm,
            embeddings=self.embedding_model,
            metrics=self.metrics
        )
        return result.to_pandas().to_dict('records')[0]
    
    async def aevaluate(self,question:str,answer:str,context:list[str])->dict[str,Any]:
        logger.info(f"Starting evluation for question {question[:100]}....")
        start_time=time.time()
        try:
            # prepare datasets for Ragas
            dataset=self.prepare_dataset(question,answer,context)

            # Run evaluation in thread pool to avoid blocking event loop
            result=await asyncio.to_thread(
                self._evaluate_with_timeout,    #callable
                dataset                         #argument
            )

            evaluation_time_ms = (time.time() - start_time) * 1000

            scores={
                "faithfulness": (float(result["_faithfulness"]) if _faithfulness in result else None),
                "answer_relevance":(float(result["_answer_relevance"]) if _answer_relevance in result else None),
                "evaluation_time_ms": round(evaluation_time_ms,2),
                "error":None
            }

            logger.info(
                    f"Evaluation completed - "
                    f"faithfulness={scores['faithfulness']}, "
                    f"answer_relevancy={scores['answer_relevancy']}, "
                    f"time={scores['evaluation_time_ms']}ms"
                )
            
            return scores
        except Exception as e:
            logger.warning(f"The evaluation failed with eception {e}")
            return self._handle_evaluation_error(e)

        



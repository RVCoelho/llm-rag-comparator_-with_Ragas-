from typing import Any
from domain.services.logging_service import RAGLogger
import time

class LLMService:
    """Serviço para consultas diretas ao LLM (sem RAG)"""
    
    def __init__(self, llm):
        self.llm = llm
        self.logger = RAGLogger()
    
    def _extract_text(self, response: Any) -> str:
        """Extrai texto de AIMessage, dict ou string"""
        if isinstance(response, str):
            return response
        elif hasattr(response, 'content'):  # AIMessage
            return response.content
        elif isinstance(response, dict) and 'answer' in response:
            return response['answer']
        elif isinstance(response, dict) and 'content' in response:
            return response['content']
        elif isinstance(response, dict) and 'text' in response:
            return response['text']
        else:
            return str(response)
    
    def answer_question(self, question):
        start_time = time.time()
        
        try:
            response = self.llm.invoke(question)
            processing_time = time.time() - start_time
            
            self.logger.log_query_execution(
                question=question,
                method="llm_only",
                response_len=len(response),
                time_ms=processing_time
            )
            
            return response
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.log_error("LLMQueryError", str(e), f"Question: {question[:100]}...")
            raise

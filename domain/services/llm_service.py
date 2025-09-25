import time
from domain.services.logging_service import RAGLogger

class LLMService:
    def __init__(self, llm):
        self.llm = llm
        self.logger = RAGLogger()
    
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

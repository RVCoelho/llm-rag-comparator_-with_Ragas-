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
    
    def answer_question(self, question: str) -> str:
        """
        Responde pergunta usando apenas LLM (sem contexto de documentos)
        
        Args:
            question: Pergunta do usuário
            
        Returns:
            Resposta como string
        """
        start_time = time.time()
        
        try:
            self.logger.logger.info(f"[LLM] Processando: {question[:50]}...")
            
            # Criar prompt simples
            prompt = f"""Responda a seguinte pergunta de forma clara e objetiva:

Pergunta: {question}

Resposta:"""
            
            # Chamar LLM
            raw_response = self.llm.invoke(prompt)
            
            # Extrair texto da resposta
            answer = self._extract_text(raw_response)
            
            elapsed = time.time() - start_time
            self.logger.logger.info(
                f"[LLM] Respondido: {len(question)} chars → {len(answer)} chars ({elapsed:.3f}s)"
            )
            
            return answer
            
        except Exception as e:
            self.logger.log_error("LLMQueryError", str(e), f"Question: {question[:50]}...")
            return f"Erro ao processar pergunta: {str(e)}"

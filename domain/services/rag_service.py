from typing import Any
from domain.services.logging_service import RAGLogger
import time

class RAGService:
    """Serviço para consultas RAG"""
    
    def __init__(self, llm, retriever):
        self.llm = llm
        self.retriever = retriever
        self.logger = RAGLogger()
    
    def _extract_text(self, response: Any) -> str:
        """Extrai texto de AIMessage, dict ou string"""
        if isinstance(response, str):
            return response
        elif hasattr(response, 'content'):  # AIMessage
            return response.content
        elif isinstance(response, dict) and 'result' in response:
            return self._extract_text(response['result'])
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
        Responde pergunta com RAG e citações numeradas
        
        Args:
            question: Pergunta do usuário
            
        Returns:
            Resposta com citações [1], [2], etc.
        """
        start_time = time.time()
        
        try:
            self.logger.logger.info(f"[RAG] Processando: {question[:50]}...")
            
            # Recuperar documentos relevantes
            try:
                docs = self.retriever.invoke(question)
            except AttributeError:
                docs = self.retriever.get_relevant_documents(question)
        
            # Validar tipo
            if isinstance(docs, str):
                self.logger.log_error("RetrieverError", "Retriever returned string instead of list", f"Value: {docs[:100]}")
                return "Erro: formato inválido retornado pelo retriever"
            
            # Criar contexto numerado
            context = ""
            for i, doc in enumerate(docs, 1):
                self.logger.logger.debug(f"Doc {i} type: {type(doc)}, has page_content: {hasattr(doc, 'page_content')}")
                context += f"[{i}] {doc.page_content}\n\n"
            
            # Criar prompt com citações
            prompt = f"""Baseado nos seguintes documentos, responda a pergunta. 
Cite as fontes usando [1], [2], etc. no final da resposta.

Documentos:
{context}

Pergunta: {question}

Resposta (com citações):"""
            
            # Chamar LLM
            raw_response = self.llm.invoke(prompt)
            answer = self._extract_text(raw_response)
            
            elapsed = time.time() - start_time
            self.logger.logger.info(
                f"[RAG] Respondido: {len(question)} chars → {len(answer)} chars ({elapsed:.3f}s)"
            )
            
            return answer
            
        except Exception as e:
            self.logger.log_error("RAGQueryError", str(e), f"Question: {question[:50]}...")
            return f"Erro ao processar pergunta: {str(e)}"
    
    def answer_question_simple(self, question: str) -> str:
        """
        Responde pergunta com RAG sem citações (para RAGAS)
        
        Args:
            question: Pergunta do usuário
            
        Returns:
            Resposta sem citações
        """
        start_time = time.time()
        
        try:
            self.logger.logger.info(f"[RAG_SIMPLE] Processando: {question[:50]}...")
            
            # Recuperar documentos relevantes
            try:
                docs = self.retriever.invoke(question)
            except AttributeError:
                docs = self.retriever.get_relevant_documents(question)
            
            # Criar contexto
            context = "\n\n".join([doc.page_content for doc in docs])
            
            # Criar prompt sem citações
            prompt = f"""Baseado no seguinte contexto, responda a pergunta de forma direta e concisa.

Contexto:
{context}

Pergunta: {question}

Resposta:"""
            
            # Chamar LLM
            raw_response = self.llm.invoke(prompt)
            answer = self._extract_text(raw_response)
            
            elapsed = time.time() - start_time
            self.logger.logger.info(
                f"[QUERY] rag_simple: {len(question)} chars → {len(answer)} chars ({elapsed:.3f}s)"
            )
            
            return answer
            
        except Exception as e:
            self.logger.log_error("RAGSimpleQueryError", str(e), f"Question: {question[:50]}...")
            return f"Erro ao processar pergunta: {str(e)}"
    

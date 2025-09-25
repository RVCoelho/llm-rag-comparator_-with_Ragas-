import time
from typing import Dict, List
from langchain.chains import RetrievalQA
from domain.services.logging_service import RAGLogger
from domain.services.citation_service import CitationService

class RAGService:
    def __init__(self, llm, retriever):
        self.llm = llm
        self.retriever = retriever
        self.logger = RAGLogger()
        self.citation_service = CitationService()
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            return_source_documents=True
        )
    
    def answer_question(self, question: str) -> Dict[str, str]:
        """Resposta RAG com citações completas"""
        start_time = time.time()
        self.citation_service.reset_counter()
        
        try:
            result = self.qa_chain({"query": question})
            
            answer = result["result"]
            source_documents = result.get("source_documents", [])
            
            citations = self._create_citations_from_sources(source_documents)
            formatted_result = self.citation_service.format_response_with_citations(answer, citations)
            
            processing_time = time.time() - start_time
            
            formatted_result["metadata"] = {
                "total_sources": len(citations),
                "files_used": list(set([c.filename for c in citations])),
                "processing_time": round(processing_time, 3)
            }
            
            self._log_query_with_citations(question, formatted_result, citations, processing_time)
            
            return formatted_result
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.log_error("RAGQueryError", str(e), f"Question: {question[:100]}...")
            raise
    
    def answer_question_simple(self, question: str) -> str:
        """Versão simples que retorna apenas a resposta (sem citações)"""
        start_time = time.time()
        
        try:
            result = self.qa_chain({"query": question})
            answer = result["result"]
            
            processing_time = time.time() - start_time
            
            self.logger.log_query_execution(
                question=question,
                method="rag_simple",
                response_len=len(answer),
                time_ms=processing_time
            )
            
            return answer
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.log_error("RAGSimpleQueryError", str(e), f"Question: {question[:100]}...")
            raise
    
    def _create_citations_from_sources(self, source_documents: List) -> List:
        citations = []
        for doc in source_documents:
            citation = self.citation_service.create_citation(
                source_document=doc,
                chunk_content=doc.page_content[:200] + "..."
            )
            citations.append(citation)
        return citations
    
    def _log_query_with_citations(self, question: str, result: Dict, citations: List, processing_time: float):
        citation_summary = self.citation_service.get_citation_summary(citations)
        
        self.logger.log_query_execution(
            question=question,
            method="rag_with_citations",
            response_len=len(result["answer"]),
            time_ms=processing_time,
            chunks=len(citations)
        )
        
        self.logger.logger.info(f"[CITATIONS] {citation_summary}")

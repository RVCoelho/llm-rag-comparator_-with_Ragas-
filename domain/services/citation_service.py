from typing import List, Dict
from dataclasses import dataclass

@dataclass
class Citation:
    number: int
    filename: str
    page: int
    chunk_content: str

class CitationService:
    def __init__(self):
        self.citation_counter = 0
    
    def reset_counter(self):
        self.citation_counter = 0
    
    def create_citation(self, source_document, chunk_content: str) -> Citation:
        self.citation_counter += 1
        
        filename = self._extract_filename(source_document)
        page = self._extract_page_number(source_document)
        
        return Citation(
            number=self.citation_counter,
            filename=filename,
            page=page,
            chunk_content=chunk_content
        )
    
    def format_response_with_citations(self, response: str, citations: List[Citation]) -> Dict[str, str]:
        if not citations:
            return {
                "answer": response,
                "sources": "Nenhuma fonte utilizada."
            }
        
        formatted_response = self._add_citation_markers(response, citations)
        formatted_sources = self._format_sources_list(citations)
        
        return {
            "answer": formatted_response,
            "sources": formatted_sources
        }
    
    def _extract_filename(self, document) -> str:
        try:
            if hasattr(document, 'metadata') and 'source' in document.metadata:
                source_path = document.metadata['source']
                return source_path.split('/')[-1].split('\\')[-1]
            return "documento_desconhecido.pdf"
        except:
            return "fonte_desconhecida.pdf"
    
    def _extract_page_number(self, document) -> int:
        try:
            if hasattr(document, 'metadata') and 'page' in document.metadata:
                return document.metadata['page'] + 1
            return 1
        except:
            return 1
    
    def _add_citation_markers(self, response: str, citations: List[Citation]) -> str:
        citation_markers = [f"[{c.number}]" for c in citations]
        
        if citation_markers:
            markers_text = " " + " ".join(citation_markers)
            if response.endswith('.'):
                response = response[:-1] + markers_text + "."
            else:
                response = response + markers_text
        
        return response
    
    def _format_sources_list(self, citations: List[Citation]) -> str:
        if not citations:
            return "Nenhuma fonte encontrada."
        
        sources_lines = ["📚 Fontes utilizadas:"]
        for citation in citations:
            source_line = f"[{citation.number}] {citation.filename} - Página {citation.page}"
            sources_lines.append(source_line)
        
        return "\n".join(sources_lines)
    
    def get_citation_summary(self, citations: List[Citation]) -> Dict[str, any]:
        if not citations:
            return {"total_sources": 0, "files": []}
        
        files = list(set([c.filename for c in citations]))
        return {
            "total_sources": len(citations),
            "total_files": len(files),
            "files": files
        }

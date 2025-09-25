import logging
import json
from datetime import datetime
from typing import List
import os

class RAGLogger:
    def __init__(self, log_level=logging.INFO):
        self.logger = logging.getLogger('rag_fii')
        self.logger.setLevel(log_level)
        
        if self.logger.handlers:
            self.logger.handlers.clear()
        
        console_handler = logging.StreamHandler()
        file_handler = logging.FileHandler('rag_fii.log', encoding='utf-8')
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
    
    def log_pdf_processing_start(self, pdfs_dir: str) -> None:
        self.logger.info(f"[FAISS] Iniciando processamento de PDFs: {pdfs_dir}")
    
    def log_pdf_discovery(self, pdf_files: List[str]) -> None:
        count = len(pdf_files)
        self.logger.info(f"[FAISS] Encontrados {count} PDF(s):")
        for i, pdf_file in enumerate(pdf_files, 1):
            filename = os.path.basename(pdf_file)
            self.logger.info(f"[FAISS]   {i}. {filename}")
    
    def log_pdf_processing_success(self, pdf_file: str, pages: int, docs: int) -> None:
        filename = os.path.basename(pdf_file)
        self.logger.info(f"[FAISS] ✅ {filename} - {pages} páginas, {docs} docs")
    
    def log_pdf_processing_error(self, pdf_file: str, error: str) -> None:
        filename = os.path.basename(pdf_file)
        self.logger.error(f"[FAISS] ❌ {filename}: {error}")
    
    def log_chunking_stats(self, original: int, chunks: int, avg_size: float) -> None:
        self.logger.info(f"[FAISS] {original} docs → {chunks} chunks (avg: {avg_size:.0f} chars)")
    
    def log_faiss_creation(self, chunks: int, path: str) -> None:
        self.logger.info(f"[FAISS] Índice criado: {chunks} chunks em {path}")
    
    def log_faiss_loaded(self, path: str) -> None:
        self.logger.info(f"[FAISS] Índice carregado: {path}")
    
    def log_query_execution(self, question: str, method: str, response_len: int, time_ms: float, chunks: int = None) -> None:
        self.logger.info(f"[QUERY] {method}: {len(question)} chars → {response_len} chars ({time_ms:.3f}s)")
    
    def log_error(self, error_type: str, message: str, context: str = None) -> None:
        self.logger.error(f"[ERROR] {error_type}: {message} | Context: {context}")

import os
import glob
import time
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from domain.services.logging_service import RAGLogger

class FaissRepository:
    def __init__(self, faiss_path, pdfs_dir, openai_api_key):
        self.faiss_path = faiss_path
        self.pdfs_dir = pdfs_dir
        self.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        self.logger = RAGLogger()
    
    def load_or_create_index(self):
        if os.path.exists(self.faiss_path):
            self.logger.log_faiss_loaded(self.faiss_path)
            return FAISS.load_local(
                self.faiss_path,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
        return self._create_new_index()
    
    def _create_new_index(self):
        start_time = time.time()
        self.logger.log_pdf_processing_start(self.pdfs_dir)
        
        pdf_files = glob.glob(os.path.join(self.pdfs_dir, "*.pdf"))
        if not pdf_files:
            raise FileNotFoundError(f"Nenhum PDF em {self.pdfs_dir}")
        
        self.logger.log_pdf_discovery(pdf_files)
        
        docs = self._process_pdf_files_with_metadata(pdf_files)
        split_docs = self._split_documents_with_metadata(docs)
        vectorstore = self._create_vectorstore(split_docs)
        
        total_time = time.time() - start_time
        self.logger.logger.info(f"[FAISS] ✅ Concluído em {total_time:.2f}s")
        
        return vectorstore
    
    def _process_pdf_files_with_metadata(self, pdf_files: List[str]) -> List[Document]:
        all_docs = []
        successful = 0
        
        for pdf_file in pdf_files:
            try:
                loader = PyPDFLoader(pdf_file)
                docs = loader.load()
                
                if docs:
                    enriched_docs = self._enrich_document_metadata(docs, pdf_file)
                    all_docs.extend(enriched_docs)
                    self.logger.log_pdf_processing_success(pdf_file, len(docs), len(enriched_docs))
                    successful += 1
                else:
                    self.logger.log_pdf_processing_error(pdf_file, "Nenhum conteúdo")
                    
            except Exception as e:
                self.logger.log_pdf_processing_error(pdf_file, str(e))
                continue
        
        self.logger.logger.info(f"[FAISS] {successful}/{len(pdf_files)} PDFs processados")
        
        if not all_docs:
            raise RuntimeError("Nenhum documento extraído")
        
        return all_docs
    
    def _enrich_document_metadata(self, docs: List[Document], pdf_file: str) -> List[Document]:
        filename = os.path.basename(pdf_file)
        enriched_docs = []
        
        for doc in docs:
            enhanced_metadata = doc.metadata.copy()
            enhanced_metadata.update({
                'source_file': filename,
                'full_path': pdf_file,
                'processed_at': time.strftime('%Y-%m-%d %H:%M:%S')
            })
            
            enriched_doc = Document(
                page_content=doc.page_content,
                metadata=enhanced_metadata
            )
            enriched_docs.append(enriched_doc)
        
        return enriched_docs
    
    def _split_documents_with_metadata(self, docs: List[Document]) -> List[Document]:
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        split_docs = splitter.split_documents(docs)
        
        for i, doc in enumerate(split_docs):
            doc.metadata['chunk_id'] = i
        
        total_chars = sum(len(doc.page_content) for doc in split_docs)
        avg_chunk_size = total_chars / len(split_docs) if split_docs else 0
        
        self.logger.log_chunking_stats(len(docs), len(split_docs), avg_chunk_size)
        
        return split_docs
    
    def _create_vectorstore(self, split_docs: List[Document]):
        vectorstore = FAISS.from_documents(split_docs, self.embeddings)
        vectorstore.save_local(self.faiss_path)
        self.logger.log_faiss_creation(len(split_docs), self.faiss_path)
        return vectorstore

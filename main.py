import os
from flask import Flask
from flask_cors import CORS
from dotenv import load_dotenv
from langchain_community.llms import OpenAI

from data.repositories.local.faiss_repository import FaissRepository
from domain.services.rag_service import RAGService
from domain.services.llm_service import LLMService
from domain.services.evaluation_service import RAGEvaluationService
from controllers.api_controller import api_bp, create_routes

# Load .env
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Config
PDFS_DIR = "assets"
FAISS_PATH = "pdf_faiss_index"

# App
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ["http://localhost:8080", "http://127.0.0.1:8080"]}}, supports_credentials=True)

# Infra
faiss_repo = FaissRepository(FAISS_PATH, PDFS_DIR, openai_api_key)
vectorstore = faiss_repo.load_or_create_index()

# Domain Services
llm = OpenAI(openai_api_key=openai_api_key)
rag_service = RAGService(llm, vectorstore.as_retriever())
llm_service = LLMService(llm)
evaluation_service = RAGEvaluationService(openai_api_key=openai_api_key)

# Controllers
create_routes(rag_service, llm_service, evaluation_service)
app.register_blueprint(api_bp)

if __name__ == "__main__":
    app.run(debug=True)

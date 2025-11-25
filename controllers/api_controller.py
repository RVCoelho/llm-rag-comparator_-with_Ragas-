from flask import Blueprint, request, jsonify
import time

api_bp = Blueprint('api_bp', __name__)

def create_routes(rag_service, llm_service, evaluation_service):
    
    @api_bp.route("/llm", methods=["POST"])
    def query_llm_only():
        """
        Endpoint 1: LLM sem RAG (pode alucinar)
        
        Body: {"question": "Sua pergunta aqui"}
        """
        data = request.json or {}
        question = data.get("question")
        
        if not question:
            return jsonify({"error": "Campo 'question' obrigatório"}), 400
        
        try:
            start_time = time.time()
            answer = llm_service.answer_question(question)
            processing_time = time.time() - start_time
            
            return jsonify({
                "question": question,
                "answer": answer,
                "method": "LLM_only",
                "processing_time": round(processing_time, 3),
                "warning": "Esta resposta pode conter alucinações - não baseada em documentos"
            })
            
        except Exception as e:
            return jsonify({"error": f"LLM error: {str(e)}"}), 500
    
    @api_bp.route("/rag", methods=["POST"])
    def query_rag_with_citations():
        """
        Endpoint 2: RAG com citações/tags
        
        Body: {"question": "Sua pergunta aqui"}
        """
        data = request.json or {}
        question = data.get("question")
        
        if not question:
            return jsonify({"error": "Campo 'question' obrigatório"}), 400
        
        try:
            start_time = time.time()
            
            answer = rag_service.answer_question(question)
            processing_time = time.time() - start_time
            
            result = {
                "question": question,
                "answer": answer,  # string da resposta
                "method": "RAG_with_citations",
                "processing_time": round(processing_time, 3)
            }
            
            return jsonify(result)
            
        except Exception as e:
            return jsonify({"error": f"RAG error: {str(e)}"}), 500
    
    @api_bp.route("/evaluate", methods=["POST"])
    def evaluate_question():
        """
        Endpoint 3: RAG sem tags + Avaliação RAGAS
        
        Body: {"question": "Sua pergunta aqui"}
        
        Retorna:
        - Resposta RAG (sem citações)
        - Avaliação RAGAS da qualidade
        """
        data = request.json or {}
        question = data.get("question")
        
        if not question:
            return jsonify({"error": "Campo 'question' obrigatório"}), 400
        
        try:
            start_time = time.time()
            
            # 1. Obter resposta RAG simples (sem citações)
            rag_answer = rag_service.answer_question_simple(question)
            
            # 2. Executar avaliação RAGAS para esta pergunta específica
            evaluation_result = evaluation_service.evaluate_single_question(
                question=question,
                rag_service=rag_service,
                llm_service=llm_service
            )
            
            processing_time = time.time() - start_time
            
            return jsonify({
                "question": question,
                "rag_answer": rag_answer,
                "method": "RAG_with_evaluation",
                "processing_time": round(processing_time, 3),
                "ragas_evaluation": evaluation_result
            })
            
        except Exception as e:
            return jsonify({"error": f"Evaluation error: {str(e)}"}), 500
    
    @api_bp.route("/health", methods=["GET"])
    def health_check():
        """Status dos endpoints"""
        return jsonify({
            "status": "healthy",
            "endpoints": {
                "/llm": "LLM puro (pode alucinar)",
                "/rag": "RAG com citações/tags",
                "/evaluate": "RAG + avaliação RAGAS"
            }
        })

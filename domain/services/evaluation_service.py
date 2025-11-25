import json
import time
import os
from typing import List, Dict, Any, Optional
from datetime import datetime
from datasets import Dataset

# Imports do RAGAS (compatibilidade)
from ragas import evaluate

try:
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision
    )
except ImportError:
    try:
        from ragas.metrics import (
            Faithfulness as faithfulness,
            AnswerRelevancy as answer_relevancy,
            ContextPrecision as context_precision
        )
        faithfulness = faithfulness()
        answer_relevancy = answer_relevancy()
        context_precision = context_precision()
    except ImportError:
        raise ImportError("Erro ao importar métricas do RAGAS")

from domain.services.logging_service import RAGLogger

class RAGEvaluationService:
    """Serviço para avaliação RAGAS sem ground truth - retorna JSON"""
    
    def __init__(self, openai_api_key: str):
        self.openai_api_key = openai_api_key
        self.logger = RAGLogger()
        
        # Configurar OpenAI
        os.environ["OPENAI_API_KEY"] = openai_api_key
        
        # Métricas sem ground truth
        self.metrics = {
            'faithfulness': faithfulness,
            'answer_relevancy': answer_relevancy,
            'context_precision': context_precision
        }
    
    def _extract_text_from_response(self, response: Any) -> str:
        """Extrai texto de diferentes tipos de resposta (AIMessage, dict, str)"""
        if isinstance(response, str):
            return response
        elif hasattr(response, 'content'):
            return response.content
        elif isinstance(response, dict) and 'answer' in response:
            return response['answer']
        elif isinstance(response, dict) and 'content' in response:
            return response['content']
        else:
            return str(response)
    
    def evaluate_single_question(self, question: str, rag_service, llm_service) -> Dict[str, Any]:
        """
        Avalia uma única pergunta e retorna resultado JSON
        
        Args:
            question: Pergunta para avaliar
            rag_service: Serviço RAG
            llm_service: Serviço LLM
            
        Returns:
            Resultado da avaliação em formato JSON
        """
        start_time = time.time()
        
        try:
            self.logger.logger.info(f"[RAGAS] Avaliando pergunta: {question[:50]}...")
            
            # 1. Coletar resposta RAG com contexts
            rag_data = self._get_rag_data(rag_service, question)
            
            # 2. Coletar resposta LLM
            llm_response = llm_service.answer_question(question)
            llm_answer = self._extract_text_from_response(llm_response)
            
            # 3. Avaliar RAG com RAGAS
            rag_scores = self._evaluate_rag_response(question, rag_data)
            
            # 4. Avaliar LLM (apenas answer_relevancy)
            llm_scores = self._evaluate_llm_response(question, llm_answer)
            
            processing_time = time.time() - start_time
            
            # 5. Compilar resultado
            result = {
                "evaluation_summary": {
                    "timestamp": datetime.now().isoformat(),
                    "processing_time": round(processing_time, 3),
                    "question_length": len(question)
                },
                "rag_evaluation": {
                    # "contexts": rag_data['contexts'],
                    "scores": rag_scores,
                    "interpretation": self._interpret_rag_scores(rag_scores)
                },
                "llm_evaluation": {
                    "answer": llm_answer,
                    "scores": llm_scores,
                    "interpretation": self._interpret_llm_scores(llm_scores)
                },
                "comparison": self._compare_rag_vs_llm(rag_scores, llm_scores),
                "recommendation": self._generate_recommendation(rag_scores, llm_scores)
            }
            
            self.logger.logger.info(f"[RAGAS] Avaliação concluída em {processing_time:.2f}s")
            
            return result
            
        except Exception as e:
            self.logger.log_error("SingleQuestionEvaluationError", str(e), question[:50])
            return {
                "error": str(e),
                "question": question,
                "timestamp": datetime.now().isoformat()
            }
    
    def _get_rag_data(self, rag_service, question: str) -> Dict[str, Any]:
        """Coleta dados RAG (resposta + contexts)"""
        try:
            # Usar método simples para não ter citações
            answer = rag_service.answer_question_simple(question)
            
            # Extrair contexts usando retriever
            contexts = []
            if hasattr(rag_service, 'retriever'):
                docs = rag_service.retriever.get_relevant_documents(question)
                contexts = [doc.page_content for doc in docs]
            elif hasattr(rag_service, 'qa_chain') and hasattr(rag_service.qa_chain, 'retriever'):
                docs = rag_service.qa_chain.retriever.get_relevant_documents(question)
                contexts = [doc.page_content for doc in docs]
            
            return {
                'answer': answer,
                'contexts': contexts
            }
            
        except Exception as e:
            self.logger.log_error("RAGDataCollectionError", str(e))
            return {'answer': f"Error: {str(e)}", 'contexts': []}
    
    def _evaluate_rag_response(self, question: str, rag_data: Dict) -> Dict[str, float]:
        """Avalia resposta RAG usando RAGAS"""
        try:
            # Preparar dataset
            dataset_dict = {
                'question': [question],
                'answer': [rag_data['answer']],
                'contexts': [rag_data['contexts']],
                'ground_truth': ['']  # Vazio - sem ground truth
            }
            
            dataset = Dataset.from_dict(dataset_dict)
            
            # Executar avaliação
            result = evaluate(dataset, metrics=list(self.metrics.values()))
            
            # Extrair scores
            scores = {}
            if hasattr(result, 'to_pandas'):
                df = result.to_pandas()
                for metric_name in self.metrics.keys():
                    if metric_name in df.columns:
                        scores[metric_name] = float(df[metric_name].iloc[0])
            
            return scores
            
        except Exception as e:
            self.logger.log_error("RAGEvaluationError", str(e))
            return {metric: 0.0 for metric in self.metrics.keys()}
    
    def _evaluate_llm_response(self, question: str, llm_answer: str) -> Dict[str, float]:
        """Avalia resposta LLM (apenas answer_relevancy)"""
        try:
            # Preparar dataset
            dataset_dict = {
                'question': [question],
                'answer': [llm_answer],
                'contexts': [[]],  # Vazio para LLM
                'ground_truth': ['']  # Vazio - sem ground truth
            }
            
            dataset = Dataset.from_dict(dataset_dict)
            
            # Avaliar apenas answer_relevancy
            result = evaluate(dataset, metrics=[self.metrics['answer_relevancy']])
            
            # Extrair score
            score = 0.0
            if hasattr(result, 'to_pandas'):
                df = result.to_pandas()
                if 'answer_relevancy' in df.columns:
                    score = float(df['answer_relevancy'].iloc[0])
            
            return {'answer_relevancy': score}
            
        except Exception as e:
            self.logger.log_error("LLMEvaluationError", str(e))
            return {'answer_relevancy': 0.0}
    
    def _interpret_rag_scores(self, scores: Dict[str, float]) -> Dict[str, str]:
        """Interpreta scores RAG em linguagem natural"""
        interpretation = {}
        
        for metric, score in scores.items():
            if metric == 'faithfulness':
                if score >= 0.8:
                    interpretation[metric] = "Excelente fidelidade - resposta baseada nos documentos"
                elif score >= 0.6:
                    interpretation[metric] = "Boa fidelidade - resposta majoritariamente baseada nos documentos"
                elif score >= 0.4:
                    interpretation[metric] = "Fidelidade moderada - resposta parcialmente baseada nos documentos"
                else:
                    interpretation[metric] = "Baixa fidelidade - resposta pode conter informações não encontradas nos documentos"
            
            elif metric == 'answer_relevancy':
                if score >= 0.8:
                    interpretation[metric] = "Excelente relevância - resposta diretamente relacionada à pergunta"
                elif score >= 0.6:
                    interpretation[metric] = "Boa relevância - resposta relacionada à pergunta"
                elif score >= 0.4:
                    interpretation[metric] = "Relevância moderada - resposta parcialmente relacionada"
                else:
                    interpretation[metric] = "Baixa relevância - resposta pode não estar relacionada à pergunta"
            
            elif metric == 'context_precision':
                if score >= 0.8:
                    interpretation[metric] = "Excelente precisão - contexts muito relevantes"
                elif score >= 0.6:
                    interpretation[metric] = "Boa precisão - contexts relevantes"
                elif score >= 0.4:
                    interpretation[metric] = "Precisão moderada - alguns contexts relevantes"
                else:
                    interpretation[metric] = "Baixa precisão - poucos contexts relevantes"
        
        return interpretation
    
    def _interpret_llm_scores(self, scores: Dict[str, float]) -> Dict[str, str]:
        """Interpreta scores LLM em linguagem natural"""
        interpretation = {}
        
        for metric, score in scores.items():
            if metric == 'answer_relevancy':
                if score >= 0.8:
                    interpretation[metric] = "Excelente relevância - resposta diretamente relacionada à pergunta"
                elif score >= 0.6:
                    interpretation[metric] = "Boa relevância - resposta relacionada à pergunta"
                elif score >= 0.4:
                    interpretation[metric] = "Relevância moderada - resposta parcialmente relacionada"
                else:
                    interpretation[metric] = "Baixa relevância - resposta pode não estar relacionada à pergunta"
        
        return interpretation
    
    def _compare_rag_vs_llm(self, rag_scores: Dict[str, float], llm_scores: Dict[str, float]) -> Dict[str, str]:
        """Compara RAG vs LLM"""
        comparison = {}
        
        # Comparar answer_relevancy
        if 'answer_relevancy' in rag_scores and 'answer_relevancy' in llm_scores:
            rag_rel = rag_scores['answer_relevancy']
            llm_rel = llm_scores['answer_relevancy']
            
            if rag_rel > llm_rel:
                comparison['answer_relevancy'] = f"RAG supera LLM ({rag_rel:.3f} vs {llm_rel:.3f})"
            elif llm_rel > rag_rel:
                comparison['answer_relevancy'] = f"LLM supera RAG ({llm_rel:.3f} vs {rag_rel:.3f})"
            else:
                comparison['answer_relevancy'] = f"RAG e LLM equivalentes ({rag_rel:.3f})"
        
        # Análise geral
        if 'faithfulness' in rag_scores:
            faith_score = rag_scores['faithfulness']
            if faith_score >= 0.7:
                comparison['overall'] = "RAG oferece maior confiabilidade devido à fidelidade aos documentos"
            else:
                comparison['overall'] = "RAG pode ter problemas de fidelidade - verificar qualidade dos documentos"
        
        return comparison
    
    def _generate_recommendation(self, rag_scores: Dict[str, float], llm_scores: Dict[str, float]) -> str:
        """Gera recomendação baseada nos scores"""
        recommendations = []
        
        # Análise de fidelidade RAG
        if 'faithfulness' in rag_scores:
            faith_score = rag_scores['faithfulness']
            if faith_score < 0.6:
                recommendations.append("⚠️ Melhorar qualidade dos documentos ou ajustar chunking")
        
        # Análise de relevância
        if 'answer_relevancy' in rag_scores and 'answer_relevancy' in llm_scores:
            rag_rel = rag_scores['answer_relevancy']
            llm_rel = llm_scores['answer_relevancy']
            
            if rag_rel < 0.6:
                recommendations.append("📝 Melhorar prompt ou ajustar retriever")
            
            if llm_rel > rag_rel + 0.2:
                recommendations.append("🔍 Considerar melhorar estratégia de retrieval")
        
        # Análise de precisão de contexto
        if 'context_precision' in rag_scores:
            prec_score = rag_scores['context_precision']
            if prec_score < 0.6:
                recommendations.append("🎯 Ajustar estratégia de chunking ou embedding")
        
        if not recommendations:
            recommendations.append("✅ Sistema funcionando bem - manter configuração atual")
        
        return " | ".join(recommendations)

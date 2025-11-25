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
            raw_answer = rag_service.answer_question_simple(question)
            answer = self._extract_text_from_response(raw_answer)
            
            # Extrair contexts usando retriever
            contexts = []
            if hasattr(rag_service, 'retriever'):
                try:
                    # Tentar método novo (invoke)
                    docs = rag_service.retriever.invoke(question)
                except AttributeError:
                    # Fallback para método antigo
                    docs = rag_service.retriever.get_relevant_documents(question)
                contexts = [doc.page_content for doc in docs]
            elif hasattr(rag_service, 'qa_chain') and hasattr(rag_service.qa_chain, 'retriever'):
                try:
                    docs = rag_service.qa_chain.retriever.invoke(question)
                except AttributeError:
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
            # Validar dados antes de avaliar
            answer = rag_data.get('answer', '').strip()
            contexts = rag_data.get('contexts', [])
            
            # Verificar se resposta e contextos não estão vazios
            if not answer or len(answer) < 10:
                self.logger.logger.warning(f"[RAGAS] Resposta RAG muito curta ou vazia: '{answer}'")
                return {metric: 0.0 for metric in self.metrics.keys()}
            
            if not contexts or len(contexts) == 0:
                self.logger.logger.warning(f"[RAGAS] Nenhum contexto encontrado para RAG")
                return {metric: 0.0 for metric in self.metrics.keys()}
            
            # Limpar contextos vazios
            contexts = [ctx.strip() for ctx in contexts if ctx and len(ctx.strip()) > 10]
            
            if not contexts:
                self.logger.logger.warning(f"[RAGAS] Contextos inválidos após limpeza")
                return {metric: 0.0 for metric in self.metrics.keys()}
            
            self.logger.logger.info(f"[RAGAS] Avaliando RAG - Q:{len(question)}chars A:{len(answer)}chars C:{len(contexts)}docs")
            
            # NOVA ABORDAGEM: Avaliar cada métrica separadamente
            scores = {}
            
            # 1. Answer Relevancy
            try:
                dataset_relevancy = Dataset.from_dict({
                    'question': [question.strip()],
                    'answer': [answer],
                    'contexts': [contexts],
                })
                
                result = evaluate(dataset_relevancy, metrics=[self.metrics['answer_relevancy']])
                
                if hasattr(result, 'to_pandas'):
                    df = result.to_pandas()
                    if 'answer_relevancy' in df.columns:
                        score_value = df['answer_relevancy'].iloc[0]
                        if score_value is not None and score_value == score_value and score_value >= 0:
                            scores['answer_relevancy'] = float(score_value)
                            self.logger.logger.info(f"[RAGAS] ✓ answer_relevancy = {scores['answer_relevancy']:.3f}")
                        else:
                            scores['answer_relevancy'] = 0.0
                            self.logger.logger.warning(f"[RAGAS] ✗ answer_relevancy = NaN/invalid")
                    else:
                        scores['answer_relevancy'] = 0.0
                        self.logger.logger.warning(f"[RAGAS] ✗ answer_relevancy não encontrado no resultado")
            except Exception as e:
                self.logger.logger.error(f"[RAGAS] Erro em answer_relevancy: {str(e)}")
                scores['answer_relevancy'] = 0.0
            
            # 2. Faithfulness
            try:
                dataset_faith = Dataset.from_dict({
                    'question': [question.strip()],
                    'answer': [answer],
                    'contexts': [contexts],
                })
                
                result = evaluate(dataset_faith, metrics=[self.metrics['faithfulness']])
                
                if hasattr(result, 'to_pandas'):
                    df = result.to_pandas()
                    if 'faithfulness' in df.columns:
                        score_value = df['faithfulness'].iloc[0]
                        if score_value is not None and score_value == score_value and score_value >= 0:
                            scores['faithfulness'] = float(score_value)
                            self.logger.logger.info(f"[RAGAS] ✓ faithfulness = {scores['faithfulness']:.3f}")
                        else:
                            scores['faithfulness'] = 0.0
                            self.logger.logger.warning(f"[RAGAS] ✗ faithfulness = NaN/invalid")
                    else:
                        scores['faithfulness'] = 0.0
                        self.logger.logger.warning(f"[RAGAS] ✗ faithfulness não encontrado")
            except Exception as e:
                self.logger.logger.error(f"[RAGAS] Erro em faithfulness: {str(e)}")
                scores['faithfulness'] = 0.0
            
            # 3. Context Precision
            try:
                dataset_precision = Dataset.from_dict({
                    'question': [question.strip()],
                    'answer': [answer],
                    'contexts': [contexts],
                    'ground_truth': [answer]  # Usar a própria resposta como ground truth
                })
                
                result = evaluate(dataset_precision, metrics=[self.metrics['context_precision']])
                
                if hasattr(result, 'to_pandas'):
                    df = result.to_pandas()
                    if 'context_precision' in df.columns:
                        score_value = df['context_precision'].iloc[0]
                        if score_value is not None and score_value == score_value and score_value >= 0:
                            scores['context_precision'] = float(score_value)
                            self.logger.logger.info(f"[RAGAS] ✓ context_precision = {scores['context_precision']:.3f}")
                        else:
                            scores['context_precision'] = 0.0
                            self.logger.logger.warning(f"[RAGAS] ✗ context_precision = NaN/invalid")
                    else:
                        scores['context_precision'] = 0.0
                        self.logger.logger.warning(f"[RAGAS] ✗ context_precision não encontrado")
            except Exception as e:
                self.logger.logger.error(f"[RAGAS] Erro em context_precision: {str(e)}")
                scores['context_precision'] = 0.0
            
            # Garantir que todas as métricas existam
            for metric in self.metrics.keys():
                if metric not in scores:
                    scores[metric] = 0.0
            
            return scores
            
        except Exception as e:
            self.logger.log_error("RAGEvaluationError", str(e))
            return {metric: 0.0 for metric in self.metrics.keys()}
    
    def _evaluate_llm_response(self, question: str, llm_answer: str) -> Dict[str, float]:
        """Avalia resposta LLM (apenas answer_relevancy)"""
        try:
            # Validar resposta
            answer = llm_answer.strip()
            
            if not answer or len(answer) < 10:
                self.logger.logger.warning(f"[RAGAS] Resposta LLM muito curta ou vazia")
                return {'answer_relevancy': 0.0}
            
            self.logger.logger.info(f"[RAGAS] Avaliando LLM - Q:{len(question)}chars A:{len(answer)}chars")
            
            # Preparar dataset (SEM contexts para LLM)
            dataset_dict = {
                'question': [question.strip()],
                'answer': [answer],
            }
            
            dataset = Dataset.from_dict(dataset_dict)
            
            # Avaliar apenas answer_relevancy
            try:
                result = evaluate(dataset, metrics=[self.metrics['answer_relevancy']])
                
                if hasattr(result, 'to_pandas'):
                    df = result.to_pandas()
                    if 'answer_relevancy' in df.columns:
                        score_value = df['answer_relevancy'].iloc[0]
                        
                        if score_value is not None and score_value == score_value and score_value >= 0:
                            score = float(score_value)
                            self.logger.logger.info(f"[RAGAS] ✓ LLM answer_relevancy = {score:.3f}")
                            return {'answer_relevancy': score}
                        else:
                            self.logger.logger.warning(f"[RAGAS] ✗ LLM answer_relevancy = NaN/invalid")
                            return {'answer_relevancy': 0.0}
                    else:
                        self.logger.logger.warning(f"[RAGAS] ✗ LLM answer_relevancy não encontrado")
                        return {'answer_relevancy': 0.0}
            except Exception as e:
                self.logger.logger.error(f"[RAGAS] Erro ao avaliar LLM: {str(e)}")
                return {'answer_relevancy': 0.0}
            
            return {'answer_relevancy': 0.0}
            
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

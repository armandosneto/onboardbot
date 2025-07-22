#!/usr/bin/env python3
"""
Script de avaliação do modelo RAG usando RAGAS com SingleTurnSample.
Avalia a precisão das respostas e relevância do contexto.
"""

import json
import pandas as pd
from typing import List, Dict, Any
from ragas import evaluate
from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    answer_similarity,
    answer_correctness
)
import os
from datetime import datetime
from langchain_google_genai import ChatGoogleGenerativeAI

# Importa as funções do projeto
from src.rag_chain import ask_question_with_context, load_api_key, initialize_llm

def load_test_data(json_path: str) -> List[Dict[str, str]]:
    """Carrega os dados de teste do arquivo JSON."""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def generate_rag_responses(test_data: List[Dict[str, str]]) -> List[SingleTurnSample]:
    """
    Gera respostas usando o sistema RAG para cada pergunta no conjunto de teste.
    Retorna uma lista de SingleTurnSample para uso com RAGAS.
    """
    samples = []
    
    print(f"Gerando respostas para {len(test_data)} perguntas...")
    
    for i, item in enumerate(test_data, 1):
        question = item["question"]
        ground_truth = item["answer"]
        
        print(f"Processando pergunta {i}/{len(test_data)}: {question[:50]}...")
        
        try:
            # Chama o sistema RAG
            rag_result = ask_question_with_context(question, use_context=True)
            
            # Extrai apenas o texto dos documentos para o contexto
            context_texts = []
            for doc in rag_result["retrieved_docs"]:
                context_texts.append(doc.page_content)
            
            # Cria o SingleTurnSample
            sample = SingleTurnSample(
                user_input=question,
                response=rag_result["answer"],
                retrieved_contexts=context_texts,
                reference=ground_truth
            )
            
            samples.append(sample)
            
        except Exception as e:
            print(f"Erro ao processar pergunta {i}: {e}")
            # Adiciona uma entrada com erro para manter o índice
            sample = SingleTurnSample(
                user_input=question,
                response=f"ERRO: {str(e)}",
                retrieved_contexts=[""],
                reference=ground_truth
            )
            samples.append(sample)
    
    return samples

def evaluate_with_ragas(samples: List[SingleTurnSample]) -> Dict[str, Any]:
    """
    Avalia as respostas usando as métricas do RAGAS com SingleTurnSample.
    """
    print("Preparando dados para avaliação com RAGAS...")
    
    # Configura o LLM do Google para o RAGAS
    api_key = load_api_key()
    os.environ["GOOGLE_API_KEY"] = api_key
    
    # Inicializa o LLM do Google para uso no RAGAS
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0,
        google_api_key=api_key
    )
    
    # Cria o EvaluationDataset com os samples
    dataset = EvaluationDataset(samples=samples)
    
    print(f"Dataset criado com {len(samples)} amostras")
    
    print("Configurando métricas de avaliação...")
    
    # Define as métricas a serem calculadas
    metrics = [
        faithfulness,           # Fidelidade: a resposta é baseada no contexto?
        answer_relevancy,       # Relevância da resposta à pergunta
        context_precision,      # Precisão do contexto (documentos relevantes no topo)
        context_recall,         # Recall do contexto (todos os docs relevantes foram recuperados?)
        answer_similarity,      # Similaridade semântica entre resposta e ground truth
        answer_correctness      # Correção da resposta comparada ao ground truth
    ]
    
    print("Iniciando avaliação... (Isso pode levar alguns minutos)")
    
    try:
        # Executa a avaliação com o LLM do Google
        result = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=llm  # Especifica o LLM do Google
        )
        
        return result
    
    except Exception as e:
        print(f"Erro durante a avaliação: {e}")
        raise

def create_detailed_report(evaluation_result, original_samples: List[SingleTurnSample]) -> pd.DataFrame:
    """
    Cria um relatório detalhado com os resultados da avaliação.
    """
    print("Criando relatório detalhado...")
    
    # Converte os resultados para DataFrame
    df_scores = evaluation_result.to_pandas()
    
    # Adiciona informações originais dos samples
    for i, sample in enumerate(original_samples):
        if i < len(df_scores):
            df_scores.loc[i, 'question'] = sample.user_input
            df_scores.loc[i, 'generated_answer'] = sample.response
            df_scores.loc[i, 'ground_truth'] = sample.reference
            df_scores.loc[i, 'context_count'] = len(sample.retrieved_contexts) if sample.retrieved_contexts else 0
            
            # Adiciona um resumo do contexto (primeiros 200 caracteres)
            context_summary = ""
            if sample.retrieved_contexts and len(sample.retrieved_contexts) > 0:
                context_summary = sample.retrieved_contexts[0][:200] + "..." if len(sample.retrieved_contexts[0]) > 200 else sample.retrieved_contexts[0]
            df_scores.loc[i, 'context_summary'] = context_summary
    
    # Reordena as colunas para melhor visualização
    columns_order = [
        'question', 
        'generated_answer', 
        'ground_truth',
        'faithfulness',
        'answer_relevancy',
        'context_precision',
        'context_recall',
        'answer_similarity',
        'answer_correctness',
        'context_count',
        'context_summary'
    ]
    
    # Seleciona apenas as colunas que existem
    available_columns = [col for col in columns_order if col in df_scores.columns]
    df_scores = df_scores[available_columns]
    
    return df_scores

def save_results_to_excel(df: pd.DataFrame, filename: str = None):
    """
    Salva os resultados em um arquivo Excel com formatação.
    """
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"evaluation_results_{timestamp}.xlsx"
    
    print(f"Salvando resultados em {filename}...")
    
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        # Aba principal com resultados detalhados
        df.to_excel(writer, sheet_name='Detailed_Results', index=False)
        
        # Aba com resumo estatístico
        numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
        summary_stats = df[numeric_columns].describe()
        summary_stats.to_excel(writer, sheet_name='Summary_Statistics')
        
        # Aba com médias por métrica
        if len(numeric_columns) > 0:
            metric_averages = df[numeric_columns].mean().to_frame('Average_Score')
            metric_averages.to_excel(writer, sheet_name='Metric_Averages')
    
    print(f"Resultados salvos com sucesso em {filename}")

def main():
    """Função principal do script de avaliação."""
    print("=== Avaliação do Sistema RAG com RAGAS ===\n")
    
    # Configurações
    test_data_path = "teste.json"
    
    try:
        # 1. Carrega dados de teste
        print("1. Carregando dados de teste...")
        test_data = load_test_data(test_data_path)
        print(f"   Carregadas {len(test_data)} perguntas de teste.")
        
        # 2. Gera respostas do sistema RAG
        print("\n2. Gerando respostas do sistema RAG...")
        rag_samples = generate_rag_responses(test_data)
        
        # 3. Avalia com RAGAS
        print("\n3. Avaliando com RAGAS...")
        evaluation_result = evaluate_with_ragas(rag_samples)
        
        # 4. Cria relatório detalhado
        print("\n4. Criando relatório detalhado...")
        detailed_report = create_detailed_report(evaluation_result, rag_samples)
        
        # 5. Salva resultados
        print("\n5. Salvando resultados...")
        save_results_to_excel(detailed_report)
        
        # 6. Exibe resumo
        print("\n=== RESUMO DOS RESULTADOS ===")
        numeric_columns = detailed_report.select_dtypes(include=['float64', 'int64']).columns
        if len(numeric_columns) > 0:
            print("\nMétricas médias:")
            for col in numeric_columns:
                if col in detailed_report.columns:
                    avg_score = detailed_report[col].mean()
                    print(f"  {col}: {avg_score:.3f}")
        
        print("\nAvaliação concluída com sucesso!")
        
    except FileNotFoundError:
        print(f"Erro: Arquivo {test_data_path} não encontrado.")
        print("Certifique-se de que o arquivo teste.json está na raiz do projeto.")
    except Exception as e:
        print(f"Erro durante a avaliação: {e}")
        raise

if __name__ == "__main__":
    main()

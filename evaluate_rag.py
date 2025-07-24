#!/usr/bin/env python3
"""
Script de avaliação do modelo RAG usando RAGAS com SingleTurnSample.
Avalia a precisão das respostas e relevância do contexto.
"""

import json
import time
import pandas as pd
from typing import List, Dict, Any
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
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
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
import asyncio

# Importa as funções do projeto
from src.rag_chain import ask_question_with_context, load_api_key

config = {
    "model": "gemini-1.5-flash",  # or other model IDs
    "temperature": 0,
    "max_tokens": None
}

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
            time.sleep(1) # Adiciona um atraso para não estourar a cota da API
            
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

async def evaluate_with_ragas(samples: List[SingleTurnSample]) -> Dict[str, Any]:
    """
    Avalia as respostas usando as métricas do RAGAS com SingleTurnSample.
    """
    print("Preparando dados para avaliação com RAGAS...")
    
    # Configura o LLM do Google para o RAGAS
    api_key = load_api_key()
    os.environ["GOOGLE_API_KEY"] = api_key
    
    # Inicializa o LLM do Google para uso no RAGAS
    # Initialize with Google AI Studio
    evaluator_llm = LangchainLLMWrapper(ChatGoogleGenerativeAI(
        model=config["model"],
        temperature=config["temperature"],
        max_tokens=config["max_tokens"],
    ))

    # Inicializa os embeddings do Google
    embeddings = LangchainEmbeddingsWrapper(
        GoogleGenerativeAIEmbeddings(
            model="models/embedding-001"
        )
    )
    
    # Cria o EvaluationDataset com os samples
    dataset = EvaluationDataset(samples=samples)
    
    print(f"Dataset criado com {len(samples)} amostras")
    
    print("Configurando métricas de avaliação...")
    
    # Define as métricas a serem calculadas
    metrics = [
        faithfulness,          # Fidelidade: a resposta é baseada no contexto?
        answer_relevancy,      # Relevância da resposta à pergunta
        context_precision,     # Precisão do contexto (documentos relevantes no topo)
        context_recall,        # Recall do contexto (todos os docs relevantes foram recuperados?)
        answer_similarity,     # Similaridade semântica entre resposta e ground truth
        answer_correctness     # Correção da resposta comparada ao ground truth
    ]
    
    print("Iniciando avaliação... (Isso pode levar alguns minutos)")
    
    time.sleep(1) # Adiciona um atraso para não estourar a cota da API
    
    try:
        # MUDANÇA: Removido "await" para chamar a função síncrona
        result = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=evaluator_llm,
            embeddings=embeddings,
            raise_exceptions=False # Adicionado para evitar que o processo pare em um único erro
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

def save_answers_to_json(answers: List[Dict[str, Any]], filename: str = "answers.json"):
    """Salva as respostas geradas com contextos em um arquivo JSON."""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(answers, f, ensure_ascii=False, indent=4)
    print(f"Respostas salvas em {filename}")

def load_answers_from_json(filename: str) -> List[Dict[str, Any]]:
    """Carrega respostas com contextos de um arquivo JSON."""
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)

async def main():
    """Função principal do script de avaliação."""
    print("=== Avaliação do Sistema RAG com RAGAS v0.3.0 ===\n")

    # --- Configurações ---
    test_data_path = "teste.json"
    answers_path = "answers.json"
    # CONTROLE DE FLUXO: Ajuste o tamanho do lote para não estourar a cota da API.
    # 5 é um valor seguro para começar.
    batch_size = 1

    try:
        # --- Etapa 1: Carregar dados de teste ---
        print("1. Carregando dados de teste...")
        test_data = load_test_data(test_data_path)
        print(f"   Carregadas {len(test_data)} perguntas de teste.")

        # --- Etapa 2: Gerar ou carregar respostas do RAG ---
        if os.path.exists(answers_path):
            print(f"\n2. Carregando respostas pré-geradas de {answers_path}...")
            answers = load_answers_from_json(answers_path)
        else:
            print("\n2. Gerando respostas do sistema RAG (isso pode levar um tempo)...")
            rag_samples_for_gen = generate_rag_responses(test_data)
            answers = [
                {
                    "question": sample.user_input,
                    "ai_answer": sample.response,
                    "contexts": sample.retrieved_contexts
                }
                for sample in rag_samples_for_gen
            ]
            save_answers_to_json(answers, answers_path)

        # --- Etapa 3: Preparar o dataset completo para avaliação ---
        print("\n3. Preparando dataset completo para avaliação...")
        all_samples = [
            SingleTurnSample(
                user_input=answer["question"],
                response=answer["ai_answer"],
                retrieved_contexts=answer["contexts"],
                reference=next((item["answer"] for item in test_data if item["question"] == answer["question"]), "")
            )
            for answer in answers
        ]

        all_results_dataframes = []

        # --- Etapa 4: Loop de processamento em lotes para evitar estouro de cota ---
        print(f"\n4. Iniciando avaliação em lotes de {batch_size}...")
        num_batches = (len(all_samples) + batch_size - 1) // batch_size
        for i in range(0, len(all_samples), batch_size):
            batch_start_index = i
            batch_end_index = min(i + batch_size, len(all_samples))
            
            # Seleciona o lote atual
            batch_samples = all_samples[batch_start_index:batch_end_index]
            batch_dataset = EvaluationDataset(samples=batch_samples)
            
            print(f"   Avaliando lote {i//batch_size + 1}/{num_batches} (itens {batch_start_index + 1} a {batch_end_index})...")
            
            # Chama a função de avaliação para o lote
            batch_result = await evaluate_with_ragas(batch_dataset)
            
            if batch_result:
                all_results_dataframes.append(batch_result.to_pandas())
            
            # Pausa opcional para ser ainda mais gentil com a API
            if i + batch_size < len(all_samples):
                print("   Lote concluído. Pausando por 5 segundos para evitar sobrecarga da API...")
                await asyncio.sleep(5)

        # --- Etapa 5: Consolidar e criar o relatório detalhado ---
        print("\n5. Consolidando resultados de todos os lotes...")
        if not all_results_dataframes:
            raise Exception("Nenhum resultado foi gerado pela avaliação. Verifique as chamadas de API.")
            
        # Junta os DataFrames de todos os lotes em um só
        final_scores_df = pd.concat(all_results_dataframes, ignore_index=True)

        # Adiciona as colunas de dados originais ao relatório final, como na sua função create_detailed_report
        print("   Adicionando dados originais ao relatório final...")
        
        # Criando um DataFrame com os dados originais para garantir a ordem correta
        original_data = pd.DataFrame({
            'question': [s.user_input for s in all_samples],
            'generated_answer': [s.response for s in all_samples],
            'ground_truth': [s.reference for s in all_samples],
            'context_count': [len(s.retrieved_contexts) if s.retrieved_contexts else 0 for s in all_samples],
            'context_summary': [
                (s.retrieved_contexts[:200] + "...") if s.retrieved_contexts and s.retrieved_contexts and len(s.retrieved_contexts) > 200 else (s.retrieved_contexts if s.retrieved_contexts and s.retrieved_contexts else "")
                for s in all_samples
            ]
        })

        # Mescla os dados originais com os scores, garantindo alinhamento
        final_report_df = pd.concat([original_data, final_scores_df.drop(columns=['question', 'answer', 'contexts', 'ground_truth'], errors='ignore')], axis=1)
        
        # Reordena as colunas para melhor visualização
        columns_order = [
            'question', 'generated_answer', 'ground_truth', 'faithfulness',
            'answer_relevancy', 'context_precision', 'context_recall',
            'answer_similarity', 'answer_correctness', 'context_count', 'context_summary'
        ]
        available_columns = [col for col in columns_order if col in final_report_df.columns]
        final_report_df = final_report_df[available_columns]

        # --- Etapa 6: Salvar e exibir resultados ---
        print("\n6. Salvando resultados em arquivo Excel...")
        save_results_to_excel(final_report_df)

        print("\n=== RESUMO DOS RESULTADOS ===")
        numeric_columns = final_report_df.select_dtypes(include=['float64', 'int64']).columns
        if len(numeric_columns) > 0:
            print("\nMétricas médias:")
            for col in numeric_columns:
                if col in final_report_df.columns and col not in ['context_count']:
                    avg_score = final_report_df[col].mean()
                    print(f"   {col}: {avg_score:.3f}")

        print("\nAvaliação concluída com sucesso!")

    except FileNotFoundError:
        print(f"Erro: Arquivo {test_data_path} não encontrado.")
        print("Certifique-se de que o arquivo teste.json está na raiz do projeto.")
    except Exception as e:
        print(f"\nERRO CRÍTICO DURANTE A AVALIAÇÃO: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    asyncio.run(main())

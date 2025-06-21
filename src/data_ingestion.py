# src/data_ingestion.py
import os
import glob
from git import Repo
from dotenv import load_dotenv
import random

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    Language,
    MarkdownHeaderTextSplitter
)

# Importa as configurações do arquivo config.py
from src.config import (
    PROJECT_ROOT,
    ROCKETCHAT_REPO_PATH,
    ROCKETCHAT_REPO_URL,
    FAISS_INDEX_PATH,
    GEMINI_EMBEDDING_MODEL
)

def clone_repository():
    """Clona o repositório do Rocket.Chat se ele não existir localmente."""
    if not os.path.exists(ROCKETCHAT_REPO_PATH):
        print(f"Repositório não encontrado. Clonando de '{ROCKETCHAT_REPO_URL}'...")
        try:
            Repo.clone_from(ROCKETCHAT_REPO_URL, ROCKETCHAT_REPO_PATH)
            print("Repositório clonado com sucesso.")
        except Exception as e:
            print(f"Falha ao clonar o repositório: {e}")
            raise
    else:
        print(f"Repositório já existe em '{ROCKETCHAT_REPO_PATH}'. Pulando clonagem.")

def process_markdown_files(repo_path):
    """Carrega e processa arquivos Markdown, preservando a estrutura de cabeçalhos."""
    print("Processando arquivos Markdown...")
    md_files = glob.glob(os.path.join(repo_path, "**/*.md"), recursive=True)
    all_md_chunks = []
    
    headers_to_split_on = [("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on, strip_headers=False)

    for md_file in md_files:
        try:
            with open(md_file, 'r', encoding='utf-8') as f:
                content = f.read()
            md_chunks = markdown_splitter.split_text(content)
            for chunk in md_chunks:
                chunk.metadata['source'] = os.path.relpath(md_file, PROJECT_ROOT)
                # Adiciona cabeçalhos relevantes ao metadata, se existirem
                if hasattr(chunk, 'metadata') and 'headers' in chunk.metadata:
                    chunk.metadata['headers'] = chunk.metadata['headers']
            all_md_chunks.extend(md_chunks)
        except Exception as e:
            print(f"Aviso: Pulando arquivo Markdown com erro de leitura: {md_file}")
            pass
            
    print(f"Processados {len(all_md_chunks)} chunks de {len(md_files)} arquivos Markdown.")
    return all_md_chunks

def process_code_files(repo_path, language, extension):
    """Carrega e processa arquivos de código para uma linguagem específica."""
    print(f"Processando arquivos de código *.{extension}...")
    code_files = glob.glob(os.path.join(repo_path, f"**/*.{extension}"), recursive=True)
    all_code_chunks = []
    
    code_splitter = RecursiveCharacterTextSplitter.from_language(
        language=language, chunk_size=1200, chunk_overlap=300
    )

    for code_file in code_files:
        try:
            loader = TextLoader(code_file, encoding='utf-8')
            documents = loader.load()
            for doc in documents:
                doc.metadata['source'] = os.path.relpath(code_file, PROJECT_ROOT)
            chunks = code_splitter.split_documents(documents)
            all_code_chunks.extend(chunks)
        except Exception as e:
            print(f"Aviso: Pulando arquivo de código com erro de leitura: {code_file}")
            pass

    print(f"Processados {len(all_code_chunks)} chunks de {len(code_files)} arquivos *.{extension}.")
    return all_code_chunks

def create_and_save_faiss_index(all_chunks):
    """Cria um índice FAISS a partir dos chunks e o salva localmente."""
    if not all_chunks:
        print("Nenhum chunk para indexar. Abortando.")
        return

    print("Inicializando o modelo de embedding do Gemini...")
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model=GEMINI_EMBEDDING_MODEL)
    except Exception as e:
        print(f"Falha ao inicializar embeddings. Verifique sua GOOGLE_API_KEY: {e}")
        raise

    print(f"Criando o índice FAISS com {len(all_chunks)} chunks. Isso pode levar algum tempo...")
    try:
        vector_store = FAISS.from_documents(all_chunks, embeddings)
        print("Índice FAISS criado com sucesso.")
        
        os.makedirs(FAISS_INDEX_PATH, exist_ok=True)
        print(f"Salvando o índice em '{FAISS_INDEX_PATH}'...")
        vector_store.save_local(FAISS_INDEX_PATH)
        print("Índice salvo com sucesso.")
    except Exception as e:
        print(f"Ocorreu um erro durante a criação ou salvamento do índice FAISS: {e}")
        raise

def run_ingestion():
    """Função principal que orquestra todo o processo de ingestão de dados."""
    load_dotenv()
    
    clone_repository()
    
    md_chunks = process_markdown_files(ROCKETCHAT_REPO_PATH)
    py_chunks = process_code_files(ROCKETCHAT_REPO_PATH, Language.PYTHON, "py")
    js_chunks = process_code_files(ROCKETCHAT_REPO_PATH, Language.JS, "js")
    ts_chunks = process_code_files(ROCKETCHAT_REPO_PATH, Language.TS, "ts")
    
    all_chunks = md_chunks + py_chunks + js_chunks + ts_chunks

    # Exibir aleatoriamente alguns chunks de cada tipo
    def print_random_chunks(chunks, tipo, n=3):
        print(f"\nExemplos aleatórios de chunks [{tipo}]:")
        if not chunks:
            print("Nenhum chunk encontrado.")
            return
        exemplos = random.sample(chunks, min(n, len(chunks)))
        for i, chunk in enumerate(exemplos, 1):
            print(f"--- {tipo} Chunk {i} ---")
            if hasattr(chunk, 'page_content'):
                print(chunk.page_content[:500])  # Limita a 500 caracteres
            else:
                print(str(chunk)[:500])
            print(f"Fonte: {chunk.metadata.get('source', 'desconhecida')}")
            print()

    print_random_chunks(md_chunks, "Markdown")
    print_random_chunks(py_chunks, "Python")
    print_random_chunks(js_chunks, "JavaScript")
    print_random_chunks(ts_chunks, "TypeScript")

    create_and_save_faiss_index(all_chunks)

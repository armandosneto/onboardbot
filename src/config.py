import os

# --- Project Root ---
# Define o caminho absoluto para a raiz do projeto
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# --- Data Configuration ---
# URL do repositório a ser clonado
ROCKETCHAT_REPO_URL = "https://github.com/RocketChat/Rocket.Chat.git"
# Caminho onde o repositório será clonado
ROCKETCHAT_REPO_PATH = os.path.join(PROJECT_ROOT, "data", "rocketchat_repo")

# --- Vector Store Configuration ---
# Caminho onde o índice FAISS será salvo e carregado
FAISS_INDEX_PATH = os.path.join(PROJECT_ROOT, "vector_store", "faiss_index")

# --- Gemini Model Configuration ---
# Modelo de embedding recomendado para tarefas de recuperação de documentos [1]
GEMINI_EMBEDDING_MODEL = "models/text-embedding-004"
# Modelo de geração de texto, otimizado para velocidade e custo
GEMINI_GENERATION_MODEL = "gemini-2.5-pro"

# --- RAG Configuration ---
# Número de documentos relevantes a serem recuperados para o contexto
RETRIEVER_K = 8

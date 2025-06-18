import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from src.config import (
    FAISS_INDEX_PATH,
    GEMINI_EMBEDDING_MODEL,
    GEMINI_GENERATION_MODEL,
    RETRIEVER_K
)

def load_api_key():
    """Carrega a chave da API do Google do arquivo.env."""
    load_dotenv() # [3, 4]
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("A variável de ambiente GOOGLE_API_KEY não foi encontrada. Por favor, configure-a no arquivo.env.")
    return api_key

def initialize_llm(api_key):
    """Inicializa o modelo de geração do Gemini."""
    os.environ["GOOGLE_API_KEY"] = api_key
    llm = ChatGoogleGenerativeAI(
        model=GEMINI_GENERATION_MODEL,
        temperature=0.1,
        convert_system_message_to_human=True
    ) # [5]
    return llm

def load_vector_store(api_key):
    """Carrega o índice FAISS local."""
    os.environ["GOOGLE_API_KEY"] = api_key
    embeddings = GoogleGenerativeAIEmbeddings(model=GEMINI_EMBEDDING_MODEL)
    
    if not os.path.exists(FAISS_INDEX_PATH):
        raise FileNotFoundError(f"Diretório do índice FAISS não encontrado em '{FAISS_INDEX_PATH}'. Execute a ingestão primeiro com 'python main.py ingest'.")
        
    vector_store = FAISS.load_local(
        FAISS_INDEX_PATH, 
        embeddings,
        allow_dangerous_deserialization=True # Necessário para carregar índices salvos com LangChain
    ) # [6]
    return vector_store

def create_retriever(vector_store, k=RETRIEVER_K):
    """Cria um retriever a partir do banco de dados vetorial."""
    return vector_store.as_retriever(search_kwargs={"k": k})

def create_prompt_template():
    """Cria o template de prompt para a cadeia RAG."""
    template = """
Você é um assistente especialista na plataforma de código aberto Rocket.Chat. Sua tarefa é responder às perguntas dos usuários com base estritamente no contexto fornecido, que é extraído da base de código e da documentação oficial do Rocket.Chat.

Regras importantes:
- Responda à pergunta usando SOMENTE as informações do seguinte contexto.
- Não utilize nenhum conhecimento prévio ou externo.
- Se o contexto não contiver informações suficientes para responder à pergunta, afirme claramente: "Com base no contexto fornecido, não tenho informações suficientes para responder a esta pergunta."
- Seja conciso e direto ao ponto.
- Ao final da sua resposta, liste as fontes que você utilizou, citando o caminho do arquivo de cada trecho do contexto.

Contexto:
{context}

Pergunta:
{question}

Resposta:
"""
    prompt = ChatPromptTemplate.from_template(template)
    return prompt

def create_rag_chain(retriever, prompt, llm):
    """Monta a cadeia RAG completa usando LCEL."""
    
    def format_docs(docs):
        # Formata os documentos recuperados em uma única string, incluindo a fonte
        return "\n\n".join(f"Fonte: {doc.metadata.get('source', 'N/A')}\nConteúdo: {doc.page_content}" for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
| prompt
| llm
| StrOutputParser()
    )
    return rag_chain

def ask_question(question: str):
    """Orquestra todo o processo de resposta a uma pergunta."""
    try:
        api_key = load_api_key()
        llm = initialize_llm(api_key)
        vector_store = load_vector_store(api_key)
        retriever = create_retriever(vector_store)
        prompt = create_prompt_template()
        
        rag_chain = create_rag_chain(retriever, prompt, llm)
        
        print("Gerando resposta...")
        answer = rag_chain.invoke(question)
        
        return answer
    except Exception as e:
        return f"Ocorreu um erro ao processar sua pergunta: {e}"


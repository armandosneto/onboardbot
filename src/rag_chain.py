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
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables.history import RunnableWithMessageHistory

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
        temperature=0.2,  # Temperatura levemente maior para respostas mais detalhadas
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
Você é um assistente especialista no projeto Rocket.Chat, ajudando desenvolvedores a entender o projeto e acelerar o onboarding.

Regras:
- Responda à pergunta usando o máximo de informações relevantes do contexto abaixo.
- Se necessário, combine informações de diferentes trechos para construir uma resposta completa.
- Se o contexto não for suficiente, explique o que está faltando, mas tente sempre extrair o máximo possível.
- Sempre cite os arquivos e, se possível, os cabeçalhos ou funções de onde as informações foram retiradas.

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

        # Recupera os documentos para mostrar o prompt formatado
        docs = retriever.get_relevant_documents(question)
        context_str = "\n\n".join(f"Fonte: {doc.metadata.get('source', 'N/A')}\nConteúdo: {doc.page_content}" for doc in docs)
        prompt_str = prompt.format(context=context_str, question=question)
        print("\nPrompt utilizado:\n" + "-"*40)
        print(prompt_str)
        print("-"*40)

        print("Gerando resposta...")
        answer = rag_chain.invoke(question)
        return answer
    except Exception as e:
        return f"Ocorreu um erro ao processar sua pergunta: {e}"

def get_memory():
    """Retorna uma instância de memória de buffer de conversa."""
    return ConversationBufferMemory(return_messages=True, memory_key="chat_history")

def rewrite_question_with_history(llm, chat_history, question):
    """
    Usa o LLM para reescrever a pergunta do usuário de forma autossuficiente, considerando o histórico da conversa.
    """
    history_str = "\n".join([
    f"Usuário: {msg.content}" if msg.type == "human" else f"Bot: {msg.content}"
    for msg in chat_history
    ])
    
    rewrite_prompt_text = (
        "Dada a conversa abaixo, reescreva a última pergunta do usuário de forma totalmente autossuficiente, "
        "removendo quaisquer expressões que façam referência ao histórico, como 'Considerando a conversa anterior', 'Com base no que foi dito', etc. "
        "Faça um pequeno resumo do contexto, se necessário.\n\n"
        "A pergunta reescrita deve ser clara e compreendida por alguém que não viu o histórico\n\n"
        "Se a última pergunta já for autossuficiente, apenas a retorne.\n\n"
        f"Histórico da conversa:\n{history_str}\n\n"
        f"Última pergunta do usuário: {question}\n\n"
        "Pergunta reescrita e autossuficiente:"
    )
    response = llm.invoke(rewrite_prompt_text)
    return response.content.strip()

def get_runnable_with_history():
    """Prepara e retorna a cadeia conversacional e o objeto de memória."""
    api_key = load_api_key()
    llm = initialize_llm(api_key)
    vector_store = load_vector_store(api_key)
    retriever = create_retriever(vector_store)
    prompt = create_prompt_template()
    rag_chain = create_rag_chain(retriever, prompt, llm)
    memory = get_memory()

    def rag_with_rewrite(inputs):
        """Função que encapsula a lógica de reescrita e RAG."""
        # Carrega o histórico da memória
        chat_history = memory.load_memory_variables({}).get("chat_history",)
        question = inputs["question"]
        
        import click
        # Se houver histórico, reescreve a pergunta. Senão, usa a original.
        if not chat_history:
            rewritten_question = question
        else:
            rewritten_question = rewrite_question_with_history(llm, chat_history, question)
            click.secho(f"[Pergunta reescrita]: {rewritten_question}", fg="yellow")
        
        # Invoca a cadeia RAG com a pergunta (reescrita ou original)
        final_answer = rag_chain.invoke(rewritten_question)
        
        # Salva a pergunta ORIGINAL do usuário e a RESPOSTA FINAL na memória.
        memory.save_context({"input": question}, {"output": final_answer})
        
        return final_answer

    return rag_with_rewrite, memory


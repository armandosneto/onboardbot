import os
from dotenv import load_dotenv
from operator import itemgetter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
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
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.runnables.history import RunnableWithMessageHistory
from src.prompts import SYSTEM_PROMPT, REWRITE_PROMPT, REWRITE_PROMPT_2

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

def create_prompt_template(use_history=False):
    """Cria o template de prompt para a cadeia RAG, separando as roles."""

    # Agora o system_prompt também terá um placeholder para o CONTEXTO.
    # O contexto é apresentado como conhecimento interno do assistente.
    system_prompt = SYSTEM_PROMPT

    if use_history:
        prompt = ChatPromptTemplate.from_messages([
            # O system_prompt agora contém o {context}
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="history"),
            # A mensagem do humano contém APENAS a {question}
            ("human", "{question}")
        ])
    else:
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{question}")
        ])

    return prompt

def create_rag_chain(
    retriever,        # pode ser None
    prompt,
    llm,
    use_history: bool = False,
    use_context: bool = True
):
    def format_docs(docs):
    	# Formata os documentos recuperados em uma única string, incluindo a fonte
        formatted_docs = "\n\n".join(f"Fonte: {doc.metadata.get('source', 'N/A')}\nConteúdo: {doc.page_content}" for doc in docs)
        return formatted_docs

    # Base do dicionário de entrada
    mapping = {"question": itemgetter("question")}

    # Só adiciona context se requisitado
    if use_context and retriever:
        mapping["context"] = itemgetter("question") | retriever | format_docs

    # Só adiciona history no template se houver histórico
    if use_history:
        mapping["history"] = itemgetter("history")

    return (mapping | prompt | llm | StrOutputParser())


# def create_rag_chain(retriever, prompt, llm, use_history=False):
#     """Monta a cadeia RAG completa usando LCEL, com suporte opcional a histórico."""

#     def format_docs(docs):
#         # Formata os documentos recuperados em uma única string, incluindo a fonte
#         formatted_docs = "\n\n".join(f"Fonte: {doc.metadata.get('source', 'N/A')}\nConteúdo: {doc.page_content}" for doc in docs)
#         return formatted_docs

#     if use_history:
#         # CADEIA CORRIGIDA
#         rag_chain = (
#             {
#                 # 1. Extrai a "question" do dicionário de input e a passa para o retriever.
#                 #    O resultado (docs) é então formatado por format_docs.
#                 "context": itemgetter("prompt") | retriever | format_docs,
                
#                 # 2. Extrai a "history" do dicionário de input.
#                 "history": itemgetter("history"),
                
#                 # 3. Extrai a "question" do dicionário de input.
#                 "question": itemgetter("question"),
#             }
#             | prompt
#             | llm
#             | StrOutputParser()
#         )
#     else:
#         # A cadeia sem histórico já estava quase correta, mas vamos torná-la explícita também
#         rag_chain = (
#             {
#                 # A chave "question" é passada para o retriever para buscar o contexto
#                 "context": itemgetter("question") | retriever | format_docs,
#                 # A chave "question" é passada diretamente para o prompt
#                 "question": itemgetter("question"),
#             }
#             | prompt
#             | llm
#             | StrOutputParser()
#         )

#     return rag_chain

# def ask_question(question: str):
#     """Orquestra todo o processo de resposta a uma pergunta."""
#     try:
#         api_key = load_api_key()
#         llm = initialize_llm(api_key)
#         vector_store = load_vector_store(api_key)
#         retriever = create_retriever(vector_store)
#         prompt = create_prompt_template()
#         rag_chain = create_rag_chain(retriever, prompt, llm)

#         # Recupera os documentos para mostrar o prompt formatado
#         docs = retriever.get_relevant_documents(question)
#         context_str = "\n\n".join(f"Fonte: {doc.metadata.get('source', 'N/A')}\nConteúdo: {doc.page_content}" for doc in docs)
#         prompt_str = prompt.format(context=context_str, question=question)
#         print("\nPrompt utilizado:\n" + "-"*40)
#         print(prompt_str)
#         print("-"*40)

#         print("Gerando resposta...")
#         answer = rag_chain.invoke(question)
#         return answer
#     except Exception as e:
#         return f"Ocorreu um erro ao processar sua pergunta: {e}"

def format_docs(docs):
    """Formata os documentos recuperados em uma única string, incluindo a fonte."""
    formatted_docs = "\n\n".join(f"Fonte: {doc.metadata.get('source', 'N/A')}\nConteúdo: {doc.page_content}" for doc in docs)
    return formatted_docs

def ask_question(
    question: str,
    use_context: bool = True,
    history_window: int = 3
):
    api_key = load_api_key()
    llm     = initialize_llm(api_key)
    store   = load_vector_store(api_key)
    retriever = store.as_retriever(search_kwargs={"k": RETRIEVER_K})

    # Memória configurável
    memory = get_memory(history_window)

    # Escolhe o template sem ou com histórico
    prompt = create_prompt_template(use_history=False)
    # Cria a chain RAG, mas decide incluir ou não o retriever
    rag_chain = create_rag_chain(
        retriever if use_context else None,
        prompt,
        llm,
        use_history=False,
        use_context=use_context
    )

    # Monta o input para invocação
    inputs = {"question": question}
    if use_context:
        # obtém docs só se for usar contexto
        docs = retriever.get_relevant_documents(question)
        inputs["context"] = format_docs(docs)  # defina sua função format_docs
    # insere histórico se houver
    hist = memory.load_memory_variables({})["chat_history"]
    if hist:
        inputs["history"] = hist

    answer = rag_chain.invoke(inputs)

    memory.save_context({"input": question}, {"output": answer})
    return answer


def get_memory(window_size: int = 3):
    return ConversationBufferWindowMemory(
        memory_key="chat_history",
        k=window_size,
        return_messages=True
    )

def rewrite_question_with_history(llm, chat_history, question, prompt=REWRITE_PROMPT):
    """
    Reescreve a pergunta do usuário de forma autossuficiente, usando as melhores práticas da LangChain (LCEL)
    de forma encapsulada. É um substituto direto da versão original.
    """
    
    # 1. Define a instrução de sistema (o papel do LLM) de forma clara.
    system_prompt = prompt

    # 2. Cria o template do prompt usando as abstrações da LangChain.
    #    MessagesPlaceholder é o local onde o histórico da conversa será inserido.
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "Ultima pergunta do usuário: **{question}** Prompt autossuficiente baseada nas regras e exemplos:")
    ])
    
    # 3. Monta e invoca a "chain" em um único passo.
    #    A chain conecta o prompt, o modelo e um parser de saída para obter a string final.
    chain = prompt | llm | StrOutputParser()
    # 4. Invoca a chain com os dados e retorna o resultado.
    #    O dicionário corresponde às variáveis no prompt (`chat_history` e `question`).
    return chain.invoke({
        "chat_history": chat_history,
        "question": question
    })

def get_runnable_with_history(window_size: int = 3):    
    """Prepara e retorna a cadeia conversacional e o objeto de memória."""
    api_key = load_api_key()
    llm = initialize_llm(api_key)
    vector_store = load_vector_store(api_key)
    retriever = create_retriever(vector_store)
    prompt = create_prompt_template()
    rag_chain = create_rag_chain(retriever, prompt, llm)
    memory = get_memory(window_size)

    def rag_with_rewrite(inputs):
        """Função que encapsula a lógica de reescrita e RAG."""
        # Carrega o histórico da memória
        chat_history = memory.load_memory_variables({}).get("chat_history", [])
        question = inputs["question"]

        # Reescreve a pergunta se houver histórico
        if chat_history:
            rewritten_question = rewrite_question_with_history(llm, chat_history, question)
            print("[Pergunta reescrita]:", rewritten_question)
        else:
            rewritten_question = question

        # Invoca a cadeia RAG com a pergunta reescrita ou original
        final_answer = rag_chain.invoke({"question": rewritten_question, "history": chat_history})

        # Salva a pergunta ORIGINAL do usuário e a RESPOSTA FINAL na memória
        memory.save_context({"input": question}, {"output": final_answer})

        return final_answer

    return rag_with_rewrite, memory

def get_runnable_with_full_history(window_size: int = 3):
    """Prepara e retorna a cadeia conversacional E o objeto de memória."""
    api_key = load_api_key()
    llm = initialize_llm(api_key)
    vector_store = load_vector_store(api_key)
    retriever = create_retriever(vector_store)
    memory = get_memory(window_size)  # ← agora usa o argumento recebido
    prompt = create_prompt_template(use_history=True)
    rag_chain = create_rag_chain(retriever, prompt, llm, use_history=True)

    def rag_with_full_history(inputs):
        """Função que encapsula a lógica de RAG com histórico completo."""
        question = inputs["question"]
        chat_history = memory.load_memory_variables({}).get("chat_history", [])

        if chat_history:
            rewritten_question = rewrite_question_with_history(
                llm, chat_history, question, prompt=REWRITE_PROMPT_2
            )
            print("[Pergunta reescrita]:", rewritten_question)
        else:
            rewritten_question = question

        final_answer = rag_chain.invoke({
            "history": chat_history,
            "question": question,
            "prompt": rewritten_question
        })

        memory.save_context({"input": question}, {"output": final_answer})
        return final_answer

    return rag_with_full_history, memory

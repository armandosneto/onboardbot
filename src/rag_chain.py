import os
from dotenv import load_dotenv
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
from langchain.memory import ConversationBufferMemory
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
    """Cria o template de prompt para a cadeia RAG, separando as roles."""
    
    # Instrução de sistema, definindo o papel do assistente
    system_prompt = """Você é um assistente especialista no projeto Rocket.Chat, ajudando desenvolvedores a entender o projeto e acelerar o onboarding.

Regras:
- Responda à pergunta usando o máximo de informações relevantes do contexto abaixo.
- Se necessário, combine informações de diferentes trechos para construir uma resposta completa.
- Se o contexto não for suficiente, explique o que está faltando, mas tente sempre extrair o máximo possível.
- Sempre cite os arquivos e, se possível, os cabeçalhos ou funções de onde as informações foram retiradas."""

    # Template da mensagem humana, com as variáveis do RAG
    human_prompt = """Contexto:
{context}

Pergunta:
{question}

Resposta:"""

    # Usa from_messages para criar um prompt com papéis explícitos
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", human_prompt)
    ])
    
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
    return ConversationBufferMemory(k=8, return_messages=True, memory_key="chat_history")

def rewrite_question_with_history(llm, chat_history, question):
    """
    Reescreve a pergunta do usuário de forma autossuficiente, usando as melhores práticas da LangChain (LCEL)
    de forma encapsulada. É um substituto direto da versão original.
    """
    
    # 1. Define a instrução de sistema (o papel do LLM) de forma clara.
    system_prompt = """Você é um especialista em reescrever prompts. Sua função é analisar um histórico de conversa entre um desenvolvedor e um assistente de código e reescrever a última pergunta do desenvolvedor para que ela seja 100% autossuficiente, capturando a intenção original.

Regras Fundamentais:
- **Identifique a Intenção Real:** O mais importante é entender o que o usuário realmente quer, com base no que o bot acabou de dizer ou fazer.
- **Incorpore o Contexto:** Integre detalhes essenciais do histórico na nova pergunta. Por exemplo, se o usuário diz "e sobre X?", a nova pergunta deve ser "Qual é a informação sobre X relacionada a Y?" ou "Considerando W, Y e Z, qual a informção sobre X?".
- **Seja Direto:** A saída final não deve conter nenhuma referência à conversa anterior (ex: "com base em...", "considerando a conversa anterior...").
- **Lide com Comandos:** Se a última entrada for um comando como "faça de novo" ou "resuma", sua tarefa é descobrir o que deve ser feito de novo ou resumido e criar uma instrução completa. Foque na última ação do Bot.
- **Autossuficiência:** Se a pergunta original já for clara e completa, retorne-a sem modificações.
- **Regra de Escape:** Se a pergunta do usuário for vaga (ex: "e sobre isso?", "quem são eles?") e o histórico da conversa não fornecer NENHUM contexto relevante para resolver a ambiguidade, NÃO TENTE EXPLICAR O PROBLEMA. Apenas retorne a pergunta original do usuário sem nenhuma alteração.
- **Atenção a sua função:** Sua função é reescrever o comando/pergunta do usuário, não responder a ela. Foque em criar uma pergunta clara e autossuficiente.

Exemplo de entrada:
---
**Exemplo 1: Reutilizando o padrão da pergunta**

Histórico:
Usuário: Onde está definido o hook `useUser` no projeto?
Assistente: O hook `useUser` está definido em `apps/meteor/client/hooks/useUser.ts`.
Pergunta do Usuário: e o `usePermission`?

**Instrução Autossuficiente Gerada:**
Onde está definido o hook `usePermission` no projeto?
---
**Exemplo 2: Lidando com o comando "faça de novo"**

Histórico:
Usuário: Liste todas as rotas de API relacionadas a `teams` no arquivo `apps/meteor/app/api/server/v1/teams.ts`.
Assistente: As rotas encontradas são: `teams.create`, `teams.list`, `teams.addMembers`, `teams.removeMember`.
Pergunta do Usuário: faça de novo

**Instrução Autossuficiente Gerada:**
Liste todas as rotas de API relacionadas a `teams` no arquivo `apps/meteor/app/api/server/v1/teams.ts`.
---
**Exemplo 3: Adicionando contexto a uma pergunta específica**

Histórico:
Usuário: Me mostre o código da função `sendMessage` no `MessageService`.
Assistente: [Exibe o bloco de código da função `sendMessage`...]
Pergunta do Usuário: explique a validação de permissão nesse trecho

**Instrução Autossuficiente Gerada:**
Explique como funciona a validação de permissão na função `sendMessage` do `MessageService`.
---
**Exemplo 4: Pedido de resumo**
Histórico:
Usuário: Quais são as principais funcionalidades do Rocket.Chat?
Assistente: O Rocket.Chat oferece funcionalidades como chat em tempo real, videoconferência,
e integração com outras plataformas.
Pergunta do Usuário: Resuma isso

**Instrução Autossuficiente Gerada:**
Resuma as principais funcionalidades do Rocket.Chat, incluindo chat em tempo real, videoconferência
e integração com outras plataformas.
"""

    
    
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
        chat_history = memory.load_memory_variables({}).get("chat_history", [])
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


# OnboardBot: Chatbot Inteligente para Onboarding de Desenvolvedores

## 📖 Visão Geral

Este projeto é um protótipo de um chatbot inteligente projetado para acelerar e simplificar o processo de _onboarding_ e o trabalho diário de desenvolvedores em uma nova base de código (_codebase_).

O chatbot utiliza uma arquitetura de **Geração Aumentada por Recuperação (RAG)** para responder a perguntas em linguagem natural sobre um repositório de software. Ele usa o código-fonte e a documentação do projeto como sua única fonte de verdade, garantindo respostas precisas, contextuais e evitando as "alucinações" comuns em modelos de linguagem genéricos.

Para este TCC, o repositório do([https://github.com/RocketChat/Rocket.Chat](https://github.com/RocketChat/Rocket.Chat)) é usado como uma simulação de uma complexa base de código empresarial.

### Tecnologias Principais

- **Motor de IA:** Google Gemini (API)
- **Orquestração:** LangChain
- **Banco de Dados Vetorial:** FAISS (local)
- **Interface:** CLI (Command-Line Interface) com Click

---

## ✨ Funcionalidades

- **Respostas Baseadas em Evidências:** Responde a perguntas sobre a arquitetura, configuração e funcionalidades usando o conteúdo exato do repositório.
- **Processamento Inteligente de Código:** Utiliza divisores de texto (_text splitters_) específicos para código (`.py`, `.js`, `.ts`) e documentação (`.md`), preservando o contexto sintático e semântico. [1, 2]
- **Citação de Fontes:** Cada resposta inclui os arquivos de origem que foram usados como contexto, permitindo que o desenvolvedor verifique e aprofunde a informação.
- **Interface de Linha de Comando:** Interação simples e direta através de três comandos principais: `ingest`, `ask` e `chat`. [3]

---

## 🚀 Como Rodar o Projeto

Siga os passos abaixo para configurar e executar o chatbot em seu ambiente local.

### Pré-requisitos

- Python 3.9 ou superior
- Git instalado em sua máquina

### 1. Configuração do Ambiente

Primeiro, clone este repositório para sua máquina local (ou crie a estrutura de pastas e arquivos manualmente).

```bash
# Navegue até o diretório onde você quer salvar o projeto
git clone <URL_DO_SEU_REPOSITORIO>
cd onboardbot
```

### 2. Ambiente Virtual e Dependências

É altamente recomendado usar um ambiente virtual para isolar as dependências do projeto. [4]

```bash
# Crie um ambiente virtual chamado 'venv'
python -m venv venv

# Ative o ambiente virtual
# No Windows:
venv\Scripts\activate
# No macOS/Linux:
source venv/bin/activate

# Instale todas as bibliotecas necessárias a partir do requirements.txt
pip install -r requirements.txt
```

### 3. Configuração da Chave de API

O projeto precisa de uma chave de API do Google Gemini para funcionar.

a. **Obtenha sua chave:** Vá até o([https://ai.google.dev/gemini-api/docs/api-key?hl=pt-br](https://ai.google.dev/gemini-api/docs/api-key?hl=pt-br)) para gerar sua chave de API gratuita. [5, 6]

b. **Crie o arquivo `.env`:** Na pasta raiz do projeto, crie um arquivo chamado `.env`.

c. **Adicione a chave:** Dentro do arquivo `.env`, adicione a seguinte linha, substituindo `SUA_CHAVE_DE_API_AQUI` pela chave que você gerou: [7]

```
GOOGLE_API_KEY="SUA_CHAVE_DE_API_AQUI"
```

O arquivo `.gitignore` já está configurado para que este arquivo não seja enviado para o repositório.

### 4. Execução

A aplicação possui três comandos principais.

#### a) Ingestão de Dados (Passo único e demorado)

Este comando prepara a base de conhecimento. Ele irá clonar o repositório do Rocket.Chat, processar todos os arquivos de código e documentação, gerar os _embeddings_ e salvar tudo em um banco de dados vetorial local (na pasta `vector_store`).

**Execute este comando apenas uma vez.** O processo pode levar vários minutos.

```bash
python main.py ingest
```

#### b) Fazer uma Pergunta

Após a conclusão da ingestão, você pode fazer perguntas ao chatbot quantas vezes quiser. Lembre-se de colocar a pergunta entre aspas.

```bash
python main.py ask "Como funciona a autenticação OAuth?"
```

**Exemplos de perguntas:**

```bash
python main.py ask "Qual a função da classe Livechat?"
python main.py ask "Como eu configuro um webhook de saída?"
python main.py ask "Onde estão definidos os papéis de usuário (roles)?"
```

#### c) Sessão de Chat Interativa

O comando `chat` inicia uma sessão interativa no terminal, permitindo que você faça várias perguntas em sequência. O chatbot reescreve cada nova pergunta considerando o histórico da conversa, tornando as perguntas autossuficientes e o contexto mais claro.

Para iniciar a sessão de chat, execute:

```bash
python main.py chat
```

- Digite suas perguntas diretamente no terminal.
- Para encerrar a sessão, digite `sair` ou `exit`.

**Exemplo de uso:**

```bash
python main.py chat
```

```
Bem-vindo ao chat interativo! Digite sua pergunta ou 'sair' para encerrar.
Você: Onde estão as configurações de autenticação?
--- Resposta ---
... (resposta do bot)
----------------
Você: E como eu altero o provedor OAuth?
--- Resposta ---
... (resposta do bot considerando o histórico)
----------------
Você: sair
Sessão encerrada.
```

---

## 🛠️ Estrutura do Projeto

```
/
|--.env                # Armazena a chave da API (não versionado)
|-- requirements.txt    # Lista de dependências Python
|-- README.md           # Este arquivo
|
|-- data/               # Onde o repositório do Rocket.Chat é clonado
|
|-- vector_store/       # Onde o índice vetorial FAISS é salvo
|
|-- src/                # Pasta com o código-fonte da aplicação
| |-- __init__.py
| |-- config.py         # Configurações centrais (caminhos, modelos)
| |-- data_ingestion.py # Lógica para ingestão e indexação dos dados
| |-- rag_chain.py      # Lógica da cadeia RAG para gerar respostas
|
|-- main.py             # Ponto de entrada da CLI
```

import click
import time
from src.data_ingestion import run_ingestion
from src.rag_chain import get_runnable_with_history, get_runnable_with_full_history
from langchain_core.messages import HumanMessage, AIMessage

@click.group()
def cli():
    """
    CLI para o chatbot RAG do Rocket.Chat.
    Use 'ingest' para construir a base de conhecimento, 'ask' para perguntas únicas e 'chat' para uma sessão interativa.
    """
    pass

@cli.command()
def ingest():
    """
    Executa o pipeline de ingestão de dados:
    1. Clona o repositório do Rocket.Chat.
    2. Processa os arquivos de código e documentação.
    3. Cria e salva um índice vetorial FAISS.
    """
    click.echo("Iniciando o processo de ingestão de dados...")
    start_time = time.time()
    try:
        run_ingestion()
        end_time = time.time()
        click.secho(f"Processo de ingestão concluído com sucesso em {end_time - start_time:.2f} segundos.", fg="green")
    except Exception as e:
        click.secho(f"O processo de ingestão falhou: {e}", fg="red")

@cli.command()
@click.argument('question', type=str)
def ask(question):
    """
    Faz uma pergunta ao chatbot RAG.
    A pergunta deve ser colocada entre aspas.
    Exemplo: python main.py ask "Como funciona a autenticação?"
    """
    click.echo(f"Pergunta recebida: '{question}'")
    click.echo("Consultando a base de conhecimento e gerando uma resposta...")
    start_time = time.time()
    answer = ask_question(question)
    end_time = time.time()
    click.secho("\n--- Resposta ---", fg="cyan")
    click.echo(answer)
    click.secho("----------------", fg="cyan")
    click.echo(f"(Resposta gerada em {end_time - start_time:.2f} segundos)")

@cli.command()
def chat():
    """
    Inicia uma sessão de chat interativa com reescrita explícita da pergunta baseada no histórico.
    Digite 'sair' ou 'exit' para encerrar a sessão.
    """
    click.secho("\nBem-vindo ao chat interativo! Digite sua pergunta ou 'sair' para encerrar.", fg="yellow")
    rag_with_rewrite, memory = get_runnable_with_history()
    session_id = "default"
    while True:
        user_input = click.prompt(click.style("Você", fg="green"), type=str)
        if user_input.strip().lower() in ["sair", "exit"]:
            click.secho("Sessão encerrada.", fg="yellow")
            break
        start_time = time.time()
        try:
            response = rag_with_rewrite({"question": user_input})
            end_time = time.time()
            click.secho("\n--- Resposta ---", fg="cyan")
            click.echo(response)
            click.secho("----------------", fg="cyan")
            click.echo(f"(Resposta gerada em {end_time - start_time:.2f} segundos)\n")
        except Exception as e:
            click.secho(f"Ocorreu um erro: {e}", fg="red")

@cli.command()
def chat_full():
    """
    Inicia uma sessão de chat interativa com o histórico completo.
    Digite 'sair' ou 'exit' para encerrar a sessão.
    """
    click.secho("\nBem-vindo ao chat interativo com histórico completo! Digite sua pergunta ou 'sair' para encerrar.", fg="yellow")
    rag_with_full_history, memory = get_runnable_with_full_history()
    while True:
        user_input = click.prompt(click.style("Você", fg="green"), type=str)
        if user_input.strip().lower() in ["sair", "exit"]:
            click.secho("Sessão encerrada.", fg="yellow")
            break
        start_time = time.time()
        try:
            response = rag_with_full_history({"question": user_input})
            end_time = time.time()
            click.secho("\n--- Resposta ---", fg="cyan")
            click.echo(response)
            click.secho("----------------", fg="cyan")
            click.echo(f"(Resposta gerada em {end_time - start_time:.2f} segundos)\n")
        except Exception as e:
            click.secho(f"Ocorreu um erro: {e}", fg="red")

if __name__ == '__main__':
    cli()

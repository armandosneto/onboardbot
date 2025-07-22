import click
import time
from src.data_ingestion import run_ingestion
from src.rag_chain import (
    ask_question,
    get_runnable_with_history,
    get_runnable_with_full_history
)

@click.group()
def cli():
    """
    CLI para o chatbot RAG do Rocket.Chat.
    Use 'ingest' para construir a base de conhecimento,
    'ask' para perguntas únicas,
    'chat' para sessão interativa simplificada,
    'chat_full' para histórico completo.
    """
    pass


@cli.command()
def ingest():
    """
    Executa o pipeline de ingestão de dados:
      1. Clona o repositório.
      2. Processa Markdown e código.
      3. Cria e salva o índice FAISS.
    """
    click.echo("Iniciando ingestão de dados...")
    start = time.time()
    try:
        run_ingestion()
        elapsed = time.time() - start
        click.secho(f"Ingestão concluída em {elapsed:.2f}s.", fg="green")
    except Exception as e:
        click.secho(f"Falha na ingestão: {e}", fg="red")


@cli.command()
@click.argument('question', type=str)
def ask(question):
    """
    Faz uma pergunta única ao chatbot.
    """
    click.echo(
        f"Pergunta: '{question}'"
    )
    click.echo("Gerando resposta...")
    start = time.time()
    answer = ask_question(
        question=question,
    )
    elapsed = time.time() - start

    click.secho("\n--- Resposta ---", fg="cyan")
    click.echo(answer)
    click.secho("----------------", fg="cyan")
    click.echo(f"(Gerado em {elapsed:.2f}s)")


@cli.command()
@click.option(
    '--use-context/--no-context',
    default=True,
    help="Incluir contexto extraído do FAISS"
)
@click.option(
    '--history-window',
    default=3,
    show_default=True,
    help="Número de trocas de mensagem na memória"
)
def chat(use_context, history_window):
    """
    Sessão de chat interativa com janela de histórico limitada.
    Digite 'sair' ou 'exit' para encerrar.
    """
    click.secho(
        "Iniciando sessão de chat (histórico limitado)…", fg="yellow"
    )
    rag_with_full, memory = get_runnable_with_full_history()


    while True:
        user_input = click.prompt(
            click.style("Você", fg="green"), type=str
        )
        if user_input.strip().lower() in ["sair", "exit"]:
            click.secho("Sessão encerrada.", fg="yellow")
            break

        start = time.time()
        response = rag_with_rewrite({
            "question": user_input,
            "use_context": use_context
        })
        elapsed = time.time() - start

        click.secho("\n--- Resposta ---", fg="cyan")
        click.echo(response)
        click.secho("----------------", fg="cyan")
        click.echo(f"(Gerado em {elapsed:.2f}s)\n")


@cli.command()
@click.option(
    '--use-context/--no-context',
    default=True,
    help="Incluir contexto extraído do FAISS"
)
@click.option(
    '--history-window',
    default=3,
    show_default=True,
    help="Número de trocas de mensagem na memória"
)
def chat_full(use_context, history_window):
    """
    Sessão de chat interativa com histórico completo.
    Digite 'sair' ou 'exit' para encerrar.
    """
    click.secho(
        "Iniciando sessão de chat (histórico completo)…", fg="yellow"
    )
    rag_with_full, memory = get_runnable_with_full_history(
        window_size=history_window
    )

    while True:
        user_input = click.prompt(
            click.style("Você", fg="green"), type=str
        )
        if user_input.strip().lower() in ["sair", "exit"]:
            click.secho("Sessão encerrada.", fg="yellow")
            break

        start = time.time()
        response = rag_with_full({
            "question": user_input,
            "use_context": use_context
        })
        elapsed = time.time() - start

        click.secho("\n--- Resposta ---", fg="cyan")
        click.echo(response)
        click.secho("----------------", fg="cyan")
        click.echo(f"(Gerado em {elapsed:.2f}s)\n")


if __name__ == '__main__':
    cli()

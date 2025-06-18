import click
import time
from src.data_ingestion import run_ingestion
from src.rag_chain import ask_question

@click.group()
def cli():
    """
    CLI para o chatbot RAG do Rocket.Chat.
    Use 'ingest' para construir a base de conhecimento e 'ask' para fazer perguntas.
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

if __name__ == '__main__':
    cli()

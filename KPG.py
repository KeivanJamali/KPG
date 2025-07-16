from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.prompts import PromptTemplate
from langsmith import Client
from datetime import datetime
from pathlib import Path
import csv
import os

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
LANGSMITH_API_KEY = os.environ.get("LANGSMITH_API_KEY")

def run_llm(query: str, chat_history: list, user_id):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=OPENAI_API_KEY)
    docsearch = PineconeVectorStore(index_name="kpg", embedding=embeddings, pinecone_api_key=PINECONE_API_KEY)
    chat = ChatOpenAI(verbose=True, temperature=0, model="gpt-4.1-nano", api_key=OPENAI_API_KEY)
    client = Client(api_key=LANGSMITH_API_KEY)

    retrieval_qa_chat_prompt = client.pull_prompt("langchain-ai/retrieval-qa-chat", include_model=True)
    stuff_documents_chain = create_stuff_documents_chain(chat, retrieval_qa_chat_prompt)
    rephrase_prompt = client.pull_prompt("langchain-ai/chat-langchain-rephrase", include_model=True)
    history_aware_retriever = create_history_aware_retriever(
        llm=chat, retriever=docsearch.as_retriever(), prompt=rephrase_prompt
    )
    qa = create_retrieval_chain(retriever=history_aware_retriever,
                                combine_docs_chain=stuff_documents_chain)
    result = qa.invoke(input={"input":query, "chat_history": chat_history})
    _log_query(query=query, response=result["answer"], user_id=user_id)

    return result["answer"]

def _log_query(query, response, user_id):
    # Get current date for the filename
    # log_path = r"/home/Keivan02/mysite/log"
    log_path = Path("logs")
    log_path.mkdir(exist_ok=True)
    date_str = datetime.now().strftime("%Y-%m-%d")
    filename = Path(f"{log_path}/{date_str}.csv")

    # Get current time
    time_str = datetime.now().strftime("%H:%M:%S")

    # Open the file in append mode or create if it doesn't exist
    with open(filename, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)

        # Write header if the file is new
        if file.tell() == 0:
            writer.writerow(["time", "user_id", "query", "response"])

        # Write the data
        writer.writerow([time_str, user_id, query, response])


print(run_llm(query="Hi", chat_history=[], user_id=1))

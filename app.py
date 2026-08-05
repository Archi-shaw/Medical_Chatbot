from flask import Flask, render_template, request
from dotenv import load_dotenv
from google.api_core.exceptions import ResourceExhausted

from src.helper import download_hugging_face_embeddings
from src.prompt import system_prompt

from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

import os

app = Flask(__name__)

load_dotenv()



PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY is missing.")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY is missing.")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

INDEX_NAME = "pdf-index"

rag_chain = None


def get_rag_chain():
    global rag_chain

    if rag_chain is None:
        print("Loading embedding model...")

        embeddings = download_hugging_face_embeddings()

        print("Connecting to Pinecone...")

        docsearch = PineconeVectorStore.from_existing_index(
            index_name=INDEX_NAME,
            embedding=embeddings
        )

        retriever = docsearch.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )

        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{input}")
            ]
        )

        question_answer_chain = create_stuff_documents_chain(
            llm,
            prompt
        )

        rag_chain = create_retrieval_chain(
            retriever,
            question_answer_chain
        )

        print("RAG chain loaded successfully!")

    return rag_chain


@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/get", methods=["POST"])
def chat():
    msg = request.form["msg"]

    try:
        chain = get_rag_chain()
        response = chain.invoke({"input": msg})
        return response["answer"]

    except ResourceExhausted:
        return "Gemini API quota exceeded. Please try again later."

    except Exception as e:
        print(e)
        return f"Error: {str(e)}"



if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)
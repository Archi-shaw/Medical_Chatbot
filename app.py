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



print("Loading embedding model...")

embeddings = download_hugging_face_embeddings()

print("Connecting to Pinecone...")

vectorstore = PineconeVectorStore.from_existing_index(
    index_name=INDEX_NAME,
    embedding=embeddings
)

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

print("Loading Gemini...")

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

document_chain = create_stuff_documents_chain(
    llm,
    prompt
)

rag_chain = create_retrieval_chain(
    retriever,
    document_chain
)

print("RAG chain ready!")



@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/get", methods=["POST"])
def chat():
    msg = request.form.get("msg", "").strip()

    if not msg:
        return "Please enter a message."

    try:
        response = rag_chain.invoke({"input": msg})
        return response["answer"]

    except ResourceExhausted:
        return "Gemini API quota exceeded. Please try again later."

    except Exception as e:
        print("ERROR:", e)
        return f"Error: {e}"



if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)
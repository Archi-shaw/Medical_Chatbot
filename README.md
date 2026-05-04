# 🩺 Medical Chatbot with LLMs, LangChain & Pinecone

A complete end-to-end **Medical Chatbot** built using **LangChain, Gemini (Google AI), Pinecone, and HuggingFace embeddings**.
This chatbot can answer medical queries based on uploaded PDF documents using **Retrieval-Augmented Generation (RAG)**.

---

## 🚀 Features

* 📄 Load and process medical PDFs
* ✂️ Intelligent text chunking
* 🔍 Semantic search using Pinecone vector database
* 🤖 LLM-powered answers using Gemini (Google Generative AI)
* 🌐 Interactive web UI with Flask
* ⚡ Fast and scalable architecture

---

## 🏗️ Tech Stack

* **LLM**: Gemini (Google Generative AI)
* **Framework**: LangChain
* **Vector DB**: Pinecone
* **Embeddings**: HuggingFace (`all-MiniLM-L6-v2`)
* **Backend**: Flask
* **Frontend**: HTML, CSS, JS
* **PDF Processing**: LangChain Document Loaders

---

## 📁 Project Structure

```
Medical_Chatbot/
│
├── data/                  # PDF files
├── src/
│   ├── helper.py         # PDF loading, chunking, embeddings
│   ├── prompt.py         # Prompt template
│
├── templates/
│   └── chat.html         # Frontend UI
│
├── static/
│   └── style.css         # Styling
│
├── app.py                # Flask app
├── store_index.py        # Pinecone indexing script
├── requirements.txt
└── .env                  # API keys (not committed)
```

---

## ⚙️ Installation

### 1. Clone the repo

```bash
git clone <your-repo-url>
cd Medical_Chatbot
```

---

### 2. Create virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

---

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🔑 Environment Variables

Create a `.env` file:

```env
PINECONE_API_KEY=your_pinecone_key
GOOGLE_API_KEY=your_gemini_key
```

⚠️ Never commit `.env` to GitHub

---

## 🧠 Create Vector Index

Run:

```bash
python store_index.py
```

This will:

* Load PDFs
* Split into chunks
* Create embeddings
* Store in Pinecone

---

## ▶️ Run the Application

```bash
python app.py
```

Open browser:

```
http://127.0.0.1:8080
```

---

## 💬 Example Queries

* What is acne?
* What causes diabetes?
* Explain hypertension


## ⭐ If you like this project

Give it a star ⭐ and share!

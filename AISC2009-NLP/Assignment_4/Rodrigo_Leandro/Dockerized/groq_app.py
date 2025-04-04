from flask import Flask, request, jsonify, render_template
from langchain.chains import RetrievalQA
from langchain_community.llms import CTransformers
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from evaluate import load
#import difflib
import pandas as pd
import torch
from langchain_groq import ChatGroq




df_val= pd.read_csv("valid.csv")

bert_score = load("bertscore") 

# === Set up Flask ===
app = Flask(__name__)

# === Load FAISS DB and Retriever ===
vector_db = FAISS.load_local(
    "vectorial_faiss_db_3",
    HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"), 
    allow_dangerous_deserialization=True
)
retriever = vector_db.as_retriever(search_kwargs={"k": 2})

# === Load LLaMA model ===
#llm = CTransformers(
#    model="llama-2-7b-chat.Q4_K_M.gguf",
#    model_type="llama",
#    config={"max_new_tokens": 512, "temperature": 0.3}
#)

llm = ChatGroq(
    api_key="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",          # O usa una variable de entorno
    model="llama3-8b-8192" # Otros: llama3-70b-8192, gemma-7b-it, etc.
)

# === Create QA Chain ===
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")

# === Serve the chatbot UI ===
@app.route("/")
def home():
    return render_template("chat.html")

# === Chat endpoint ===
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    query = data.get("query", "")

    if not query:
        return jsonify({"error": "Missing 'query' parameter"}), 400

    try:
        result = qa_chain.invoke({"query": query})
        response = result.get("result", str(result))
        # Buscar respuesta esperada desde tu df_val
        row = df_val[df_val["query"].str.lower() == query.strip().lower()]
        if not row.empty:
            expected_answer = row["answers"].values[0]
        else:
            expected_answer = "Unknown"

        # Calcular BERTScore si es posible
        if expected_answer != "Unknown":
            bert_result = bert_score.compute(
                predictions=[response],
                references=[expected_answer],
                lang="en",
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
            bert_f1 = round(bert_result["f1"][0], 4)
        else:
            bert_f1 = None

        return jsonify({
            "response": response,
            "bert_score_f1": bert_f1,
            "expected_answer": expected_answer
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# === Run ===
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)

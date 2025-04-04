FROM python:3.13.2
COPY . /app             
WORKDIR /app

# Copy application files
#llama-2-7b-chat.Q4_K_M.gguf

COPY groq_app.py requirements.txt  valid.csv  /app/
COPY templates/ /app/templates/  
COPY vectorial_faiss_db_3/ /app/vectorial_faiss_db_3/  

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 5000 for Flask
EXPOSE 5000

# Run the Flask app
CMD ["python", "groq_app.py"]
#"--server.adress=0.0.0.0"
# Use an official lightweight Python image
FROM python:3.12.9

# Set the working directory inside the container
WORKDIR /app

# Copy only requirements first (better Docker caching)
COPY requirements.txt /app/

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the remaining application files
COPY app.py best_biGRU_model.pth best_model_rnn.h5 /app/
COPY templates/ /app/templates/  

# Expose port 5002 for Flask
EXPOSE 5002

# Run the Flask app
ENTRYPOINT ["python", "app.py"]

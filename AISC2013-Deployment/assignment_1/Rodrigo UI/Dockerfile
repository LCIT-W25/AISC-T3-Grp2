# Use an official lightweight Python image
FROM python:3.8

# Set the working directory inside the container
WORKDIR /app2



# Copy application files
COPY app2.py config.json requirements.txt untuned_naive_bayes.pkl naive_bayes_model.pkl untuned_svm.pkl svm_model.pkl vectorizer.pkl /app2/
COPY templates/ /app2/templates/  

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 5000 for Flask
EXPOSE 5000

# Run the Flask app
CMD ["python", "app2.py"]

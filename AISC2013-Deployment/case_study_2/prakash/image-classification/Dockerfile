# Use an official lightweight Python image
FROM python:3.12.9

# Set the working directory inside the container
WORKDIR /app2

# Copy application files
COPY app2.py requirements.txt efficientnet_model.h5 cnn_model.h5 /app2/
COPY templates/ /app2/templates/  

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt
#RUN conda env create -f environment.yml

# Expose port 5000 for Flask
EXPOSE 5000

# Run the Flask app
CMD ["python", "app2.py"]

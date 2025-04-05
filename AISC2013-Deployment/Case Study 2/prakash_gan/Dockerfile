# Use an official lightweight Python image
FROM python:3.12.9

# Set the working directory inside the container
WORKDIR /app

# Copy only requirements first (better Docker caching)
COPY requirements.txt /app/

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the remaining application files
COPY app.py best_discriminator_model.keras best_generator_model.keras index.html /app/

# Run the Flask app
ENTRYPOINT ["python", "app.py"]
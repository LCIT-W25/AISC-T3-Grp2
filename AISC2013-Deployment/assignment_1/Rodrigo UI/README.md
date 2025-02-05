**Sentiment Analysis Flask App**

This project is a Flask-based web application that implements a Sentiment Analysis model using Naïve Bayes (NB) and Support Vector Machine (SVM). The application is containerized using Docker.

📌 Prerequisites

Make sure you have the following installed:

Docker

Git

🚀 Installation & Setup

1️⃣ Clone the Repository

git clone https://github.com/yourusername/yourrepo.git
cd yourrepo

2️⃣ Build the Docker Image

Run the following command to build the Docker image:

docker build --no-cache -t sentiment-analysis .

3️⃣ Run the Docker Container

Start the Flask app by running the container:

docker run -p 5000:5000 sentiment-analysis

4️⃣ Access the Web Application

Once the container is running, open your browser and visit:

http://localhost:5000

📂 Project Structure

📦 yourrepo
 ┣ 📂 templates
 ┃ ┣ 📜 index.html  # Frontend template
 ┣ 📜 app2.py  # Flask application code
 ┣ 📜 Dockerfile  # Docker build instructions
 ┣ 📜 naive_bayes_model.pkl  # Pre-trained Naïve Bayes model
 ┣ 📜 svm_model.pkl  # Support Vector Machine model
 ┣ 📜 vectorizer.pkl  # Text vectorization model
 ┣ 📜 requirements.txt  # Python dependencies
 ┣ 📜 Readme.txt  # Additional documentation
 ┣ 📜 README.md  # Project documentation

📦 Docker Hub (Optional)

If you want to push the image to Docker Hub, run:

docker tag sentiment-analysis your-dockerhub-username/sentiment-analysis:latest

docker push your-dockerhub-username/sentiment-analysis:latest

To allow others to pull the image, they can use:

docker pull your-dockerhub-username/sentiment-analysis:latest

📜 License

This project is licensed under the MIT License.

✨ Author

Created by Rodrigo Velazquez
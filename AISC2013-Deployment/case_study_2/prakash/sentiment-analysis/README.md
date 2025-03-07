🚀 Installation & Setup

1️⃣ Build the Docker Image

Run the following command to build the Docker image:

docker build --no-cache -t sentiment-docker .

2️⃣ Run the Docker Container

Start the Flask app by running the container:

docker run -p 5002:5002 sentiment-docker

3️⃣ Access the Web Application

Once the container is running, open your browser and visit:

http://localhost:5002


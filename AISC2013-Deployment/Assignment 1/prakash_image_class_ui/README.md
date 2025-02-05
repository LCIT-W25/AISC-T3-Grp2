# Flask APP for Image Classification

## Steps to Run the Flask App

1. **Create and activate a virtual environment**:

   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

2. **Install the required dependencies**:

   ```sh
   pip install -r requirements.txt
   ```

3. **Run the Flask app**:

   ```sh
   python3 run.py # On windows use `python run.py`
   or
   flask run
   ```

The app should now be running at `http://127.0.0.1:5000/`.

4. **Build the Docker image**:

   ```sh
   docker build -t flask-app .
   ```

5. **Run the Docker container**

   ```sh
   docker run -p 5000:5000 flask-app
   ```

## Project Structure

```bash
prakash_image_class_ui/
    ├── .gitignore
    ├── app/
    │   ├── __init__.py
    │   ├── model.py
    │   ├── routes.py
    │   ├── static/
    │   │   ├── scripts.js
    │   │   ├── styles.css
    │   │   └── uploads/
    │   └── templates/
    │       └── index.html
    ├── config.py
    ├── models/
    │   ├── dnn_model.pth
    │   ├── knn_model.pkl
    │   ├── label_encoder_dnn.pkl
    │   └── label_encoder_knn.pkl
    ├── README.md
    ├── requirements.txt
    ├── run.py
    └── venv/ # virtual env
```

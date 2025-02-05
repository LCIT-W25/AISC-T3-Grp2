import os


class Config:
    # Set via env variable in production
    SECRET_KEY = os.environ.get('SECRET_KEY', 'alongsecretekeyhahahah')
    FLASK_ENV = os.environ.get('FLASK_ENV', 'development')
    UPLOAD_FOLDER = 'app/static/uploads'
    ALLOWED_EXTENSIONS = {'txt', 'csv', '.png', '.jpg', '.jpeg'}

from flask import Flask

app = Flask(__name__)
app.config['WTF_CSRF_ENABLED'] = True
app.config.from_object('config.Config')

from app import routes

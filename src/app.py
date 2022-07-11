"""
application: Image Classification Model Playground

author: Prajas Kadepurkar | Elton Lemos | Abhishek Ghoshal | Tanmey Saraiya

"""

from flask import Flask

app = Flask(__name__)
app.config.from_pyfile("config.py")

#Importing and Registering Blueprints
from flask import Blueprint

from main.routes import main
app.register_blueprint(main)

from auth.routes import auth
app.register_blueprint(auth)

#Run Server
if __name__ == "__main__":
    app.run(host = "0.0.0.0")
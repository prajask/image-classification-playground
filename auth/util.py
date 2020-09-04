"""
Authentication Utility Functions

"""

from flask import session

from extensions import database

from config import SECRET_KEY

from uuid import uuid4
from binascii import hexlify
from hashlib import pbkdf2_hmac

def set_user_email(email):
    if not get_user(email):
        session["user_email"] = email
        return True
    
    else:
        return False

def set_user_password(password):
    session["user_password"] = create_password_hash(password)

def sign_up(user_name):
    user_id = uuid4().hex

    user = {
        "name": f"{user_name['first_name']} {user_name['last_name']}",
        "email_id": session["user_email"],
        "password_hash": session["user_password"]
    }

    database.collection("users").document(user_id).set(user)

    del session["user_email"]
    del session["user_password"]

    user["id"] = user_id
    set_user_session(user)

def login(user_credentials):
    user = verify_user(user_credentials)
    if user:
        set_user_session(user)

        return True

    else:
        return False

def verify_user(user_credentials):
    user = get_user(user_credentials["email"])

    return user if user and verify_password(user_credentials["password"], user["password_hash"]) else False

def get_user(email):
    user_query = database.collection("users").where("email_id", "==", email).get()
    
    if user_query:
        user = user_query[0].to_dict()
        user["id"] = user_query[0].id

        return user

    else:
        return False

def verify_password(password, password_hash):
    return create_password_hash(password) == password_hash

def set_user_session(user):
    session["logged_in"] = True
    session["user"] = user
    
    del session["user"]["password_hash"]

def create_password_hash(password):
    return str( hexlify( pbkdf2_hmac( "sha256",  password.encode(), SECRET_KEY.encode(), 5000) ).decode() )

def logout():
    del session["user"]
    session["logged_in"] = False
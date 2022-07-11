"""
Authentication Routes

"""


from flask import Blueprint, render_template, redirect, request, url_for, session
from auth import util

auth = Blueprint("auth", __name__, url_prefix = "/auth", template_folder = "templates")

@auth.route("/login", methods = ["GET", "POST"])
def login():
    if "logged_in" in session and session["logged_in"]:
        return redirect(url_for( "main.index" ))

    else:
        if request.method == "POST":
            if util.login(request.form):
                return redirect(url_for( "main.index" ))

            else:
                return render_template("auth/login.html", error = True)
        
        else:
            return render_template("auth/login.html")

@auth.route("/sign_up", methods = ["GET", "POST"])
def sign_up():
    if "logged_in" in session and session["logged_in"]:
        return redirect(url_for( "main.index" ))

    else:
        if request.method == "POST":
            if util.set_user_email(request.form["email"]):
                return redirect(url_for( "auth.sign_up_password" ))

            else:
                return render_template("auth/sign_up_email.html", error="already_registered")
        
        else:
            return render_template("auth/sign_up_email.html")

@auth.route("/sign_up/password", methods = ["GET", "POST"])
def sign_up_password():
    if request.method == "POST":
        util.set_user_password(request.form["password"])

        return redirect(url_for( "auth.sign_up_name" ))

    else:
        return render_template("auth/sign_up_password.html")

@auth.route("/sign_up/name", methods = ["GET", "POST"])
def sign_up_name():
    if request.method == "POST":
        util.sign_up(request.form)

        return redirect(url_for( "main.index" ))

    else:
        return render_template("auth/sign_up_name.html")

@auth.route("/forgot_password", methods = ["GET", "POST"])
def forgot_password():
    return "Forgot Password"

@auth.route("logout")
def logout():
    util.logout()

    return redirect(url_for( 'main.index' ))
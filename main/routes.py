"""
Main App Routes

"""

from flask import Blueprint, render_template, redirect, request, url_for, session
from main import util

main = Blueprint("main", __name__, template_folder = "templates")

@main.route("/")
def index():
    return render_template("/main/index.html")

@main.route("/contributors")
def contributors():
    return render_template("/main/contributors.html")

@main.route("/user_models")
def user_models():
    if "logged_in" in session and session["logged_in"]:
        return render_template("/main/user_models.html", models = util.get_user_models())

    else:
        return redirect(url_for( "auth.login" ))

@main.route("/upload_images", methods = ["GET", "POST"])
def upload_images():
    if "id" not in session or not session["id"]:
        util.create_session()

    if request.method == "POST":
        util.upload_images(request.files.getlist("image-file-input"))

        return redirect(url_for( "main.select_model" ))

    else:
        return render_template("/main/upload_images.html")

@main.route("/select_model")
def select_model():
    if not "images_uploaded" in session or not session["images_uploaded"]:
        return redirect(url_for( "main.upload_images" ))

    else:
        if "logged_in" in session and session["logged_in"]:
            return render_template("/main/select_model.html", models = util.get_user_models())
        
        else:
            return redirect(url_for( "main.classifying_images", model = "default" ))


@main.route("/classifying_images/<string:model>")
def classifying_images(model):
    if (not "images_uploaded" in session or not session["images_uploaded"]):
        return redirect(url_for( "main.upload_images" ))
    
    else:
        return render_template("main/classifying_images.html", model = model)

@main.route("/classify_images/<string:model>")
def classify_images(model):
    util.classify_images(model)
    
    return "Done"

@main.route("/classified_images")
def classified_images():
    return render_template("main/classified_images.html")

@main.route("/record_corrections", methods = ["POST"])
def record_corrections():
    util.record_corrections(request.form)

    return redirect(url_for( "main.set_prediction_accuracy" ))

@main.route("/set_prediction_accuracy")
def set_prediction_accuracy():
    util.set_prediction_accuracy()

    return redirect(url_for( "main.model_insights" ))

@main.route("/model_insights")
def model_insights():   
    return render_template("/main/model_insights.html")

@main.route("/model_insights_from_id/<string:model_id>")
def model_insights_from_id(model_id):
    util.get_model_insights_from_cloud(model_id) if model_id != "default" else False

    util.set_model_path(model_id)

    return render_template("/main/model_insights.html")

@main.route("/tune_model", methods=["GET", "POST"])
def tune_model():
    if "logged_in" in session and session["logged_in"]:
        if request.method == "POST":
            util.set_model_layers(request.form)

            return render_template("main/tuning_model.html")
        
        else:
            from main.util import ACTIVATION_FUNCTIONS
            from main.util import OPTIMIZATION_FUNCTIONS
            from main.util import LOSS_FUNCTIONS

            return render_template("/main/tune_model.html", activation_functions = ACTIVATION_FUNCTIONS, optimization_functions = OPTIMIZATION_FUNCTIONS, loss_functions = LOSS_FUNCTIONS)

    else:
        return redirect(url_for( "auth.login" ))

@main.route("/train_model")
def train_model():
    model, trained_model = util.train_model()
    
    util.save_model_local(model)
    util.save_model_accuracy_local(trained_model)
    util.save_model_loss_local(trained_model)

    return "Done"

@main.route("/trained_model")
def trained_model():
    return render_template("/main/trained_model.html")

@main.route("/saving_model")
def saving_model():
    if "predictions" in session:
        del session["predictions"]

    return render_template("/main/saving_model.html")

@main.route("/save_model")
def save_model():
    util.save_model_to_cloud()

    util.finish_classification()
    
    return "Done"

@main.route("/finish_classification")
def finish_classification():
    util.finish_classification()

    return redirect(url_for( "main.index" ))
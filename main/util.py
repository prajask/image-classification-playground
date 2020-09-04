"""
Main App Utility Functions

"""

from flask import session
from config import UPLOADS_FOLDER, CORRECTIONS_FOLDER

from uuid import uuid4
from os import mkdir, path, listdir
from shutil import rmtree, move
import json

from extensions import database, cloud_storage

import cv2
import tensorflow as tf
import tensorflow.keras.layers as Layers
import tensorflow.keras.activations as Activations
import tensorflow.keras.models as Models
import tensorflow.keras.optimizers as Optimizer
import tensorflow.keras.metrics as Metrics
import tensorflow.keras.utils as Utils
from keras.utils.vis_utils import model_to_dot
import numpy as np
import matplotlib.pyplot as plot

CLASS_LABLES = ['Buildings','Forest','Glacier','Mountain','Sea','Street']
ACTIVATION_FUNCTIONS = ["relu", "elu","selu", "softmax", "softplus", "softsign", "tanh", "sigmoid", "hard_sigmoid", "exponential", "linear"]
OPTIMIZATION_FUNCTIONS = ["Adam","SGD","RMSprop","Adagrad","Adadelta","Adamax","Nadam"]
LOSS_FUNCTIONS = ["sparse_categorical_crossentropy","mean_absolute_error","mean_absolute_percentage_error","mean_squared_logarithmic_error","squared_hinge","hinge","categorical_hinge","logcosh","huber_loss","categorical_crossentropy","mean_squared_error","binary_crossentropy", "kullback_leibler_divergence","poisson","cosine_proximity","is_categorical_crossentropy"]

def create_session():
    session["id"] = uuid4().hex

def get_user_models():
    models = dict()

    models_query = database.collection("users").document(session["user"]["id"]).collection("models").get()

    for model in models_query:
        current_model = model.to_dict()
        models[model.id] = current_model["name"]

    return models

def upload_images(images):
    upload_path = get_upload_path()
    create_path_if_not_exists(upload_path)
    
    for image in images:
        image.save(f"{upload_path}/{image.filename}")

    session["images_uploaded"] = True
    session["number_of_images"] = len(images)

def get_upload_path():
    upload_path = f"{UPLOADS_FOLDER}/temp/{session['id']}"

    return upload_path

def create_path_if_not_exists(upload_path):
    mkdir(upload_path) if not path.exists(upload_path) else False

def classify_images(model_id):
    model = get_model(model_id)

    set_model_path(model_id)

    set_model_summary(model)

    images = read_images()

    session["predictions"] = get_predictions(images, model)

def get_model(model_id):
    if model_id == "default":
        return Models.load_model("static/models/default/model.h5")

    else:
        get_model_from_cloud(model_id)

        get_model_insights_from_cloud(model_id)

        return Models.load_model( f"static/models/temp/{session['id']}/{model_id}/model.h5" )

def get_model_from_cloud(model_id):
    create_path_if_not_exists(f"static/models/temp/{session['id']}")
    create_path_if_not_exists(f"static/models/temp/{session['id']}/{model_id}")

    model_blob = cloud_storage.blob( f"{session['user']['id']}/models/{model_id}/model.h5" )
    model_blob.download_to_filename( f"static/models/temp/{session['id']}/{model_id}/model.h5" )

def get_model_insights_from_cloud(model_id):
    create_path_if_not_exists(f"static/models/temp/{session['id']}")
    create_path_if_not_exists(f"static/models/temp/{session['id']}/{model_id}")

    model_accuracy_image_blob = cloud_storage.blob( f"{session['user']['id']}/models/{model_id}/model_accuracy.png" )
    model_accuracy_image_blob.download_to_filename( f"static/models/temp/{session['id']}/{model_id}/model_accuracy.png" )

    model_validation_image_blob = cloud_storage.blob( f"{session['user']['id']}/models/{model_id}/model_loss.png" )
    model_validation_image_blob.download_to_filename( f"static/models/temp/{session['id']}/{model_id}/model_loss.png" )

def set_model_path(model_id):
    if model_id == "default":
        session["model_path"] = f"models/default/"
        
    else:
        session["model_path"] = f"models/temp/{session['id']}/{model_id}"

def set_model_summary(model):
    model_summary = json.loads(model.to_json())

    layers = model_summary["config"]["layers"]
    
    input_layer = layers[1]

    session["model_summary"] = {
        "input_layer": {
            "class": "Conv2D",
            "class_name": "Convolution",
            "input_size": input_layer["config"]["batch_input_shape"][1],
            "filter_size": input_layer["config"]["filters"],
            "kernel_size": input_layer["config"]["kernel_size"][0],
            "activation": input_layer["config"]["activation"]
        }
    }

    session["model_summary"]["layers_before_flatten"] = list()
    session["model_summary"]["layers_after_flatten"] = list()

    before_flatten = True

    for layer_number in range(2, (len(layers) - 1)):
        current_layer = layers[layer_number]
            

        if current_layer["class_name"] == "Conv2D":
            session["model_summary"]["layers_before_flatten"].append({
                "number": f"{layer_number}",
                "class": "Conv2D",
                "class_name": "Convolution",
                "filter_size": current_layer["config"]["filters"],
                "kernel_size": current_layer["config"]["kernel_size"][0],
                "activation": current_layer["config"]["activation"]
            })

        elif current_layer["class_name"] == "MaxPooling2D":
            session["model_summary"]["layers_before_flatten"].append({
                "number": f"{layer_number}",
                "class": "MaxPool2D",
                "class_name": "Max Pooling",
                "pool_size": current_layer["config"]["pool_size"][0]
            })

        elif current_layer["class_name"] == "Flatten":
            session["model_summary"]["flatten_layer"] = {
                "number": f"{layer_number}",
                "class": "Flatten",
                "class_name": "Flatten"
            }

            before_flatten = False

        elif current_layer["class_name"] == "Dense":
            session["model_summary"]["layers_after_flatten"].append({
                "number": f"{layer_number}",
                "class": "Dense",
                "class_name": "Dense",
                "units": current_layer["config"]["units"],
                "activation": current_layer["config"]["activation"]
            })

        elif current_layer["class_name"] == "Dropout":
            if before_flatten:
                session["model_summary"]["layers_before_flatten"].append({
                    "number": f"{layer_number}",
                    "class": "Dropout",
                    "class_name": "Dropout",
                    "rate": current_layer["config"]["rate"]
                })

            else:
                session["model_summary"]["layers_after_flatten"].append({
                    "number": f"{layer_number}",
                    "class": "Dropout",
                    "class_name": "Dropout",
                    "rate": current_layer["config"]["rate"]
                })

    output_layer = layers[-1]

    session["model_summary"]["output_layer"] = {
        "name": "output_layer",
        "class": "Dense",
        "class_name": "Dense",
        "units": output_layer["config"]["units"],
        "activation": output_layer["config"]["activation"]
    }

def read_images():
    images = dict()

    user_uploads = f"{UPLOADS_FOLDER}/temp/{session['id']}"

    for image_name in listdir(user_uploads):
        image = cv2.imread(f"{user_uploads}/{image_name}")
        casted_image = tf.cast(image, tf.float32)

        images[image_name] = image

    return images

def get_predictions(images, model):
    predictions = dict()

    for image in images:
        image_label = CLASS_LABLES[ model.predict_classes( np.expand_dims( images[image], axis = 0 ) )[0] ]

        predictions[image] = image_label

    return predictions

def record_corrections(user_corrections):
    corrections = dict()

    for image in user_corrections:
        if user_corrections[image] != session["predictions"][image]:
            corrections[image] = {
                "prediction": session["predictions"][image],
                "correction": user_corrections[image]
            }

    if corrections:
        create_correction_file(corrections)

        save_correction_images(corrections)

    session["number_of_corrections"] = len(corrections)

def create_correction_file(corrections):
    corrections_path = f"{CORRECTIONS_FOLDER}/{session['id']}"
    
    create_path_if_not_exists(corrections_path)
    
    with open(f"{corrections_path}/corrections.json", "a+") as corrections_file:
        json.dump(corrections, corrections_file)

def save_correction_images(corrections):
    images_path = f"{UPLOADS_FOLDER}/temp/{session['id']}"
    corrections_path = f"{CORRECTIONS_FOLDER}/{session['id']}"

    for image in corrections:
        try:
            move(f"{images_path}/{image}", corrections_path)
        except:
            continue

def set_prediction_accuracy():
    session["prediction_accuracy"] = int ( ( ( session["number_of_images"] - session["number_of_corrections"] ) / session["number_of_images"] ) * 100 )

def set_model_layers(model_parameters):

    session["model_layers"] = create_model_layers(model_parameters)
    session["model_training_parameters"] = get_training_parameters(model_parameters)

def train_model():
    model = create_model(session["model_layers"])

    model_training_parameters = session["model_training_parameters"]

    optimizer = get_optimizer(model_training_parameters["optimization_function"], model_training_parameters["learning_rate"])

    model.compile( optimizer = optimizer, loss = model_training_parameters["loss_function"], metrics = ["accuracy"] )

    train_images = np.load( "static/model_tuning/numpy_files/train_images.npy" )[0 : 100]
    train_labels = np.load( "static/model_tuning/numpy_files/train_labels.npy" )[0 : 100]
    test_images = np.load( "static/model_tuning/numpy_files/test_images.npy" )[0 : 100]
    test_labels = np.load( "static/model_tuning/numpy_files/test_labels.npy" )[0 : 100]

    trained_model = model.fit(
        train_images,
        train_labels,
        epochs = model_training_parameters["number_of_epochs"],
        validation_data = ( test_images, test_labels ),
        verbose = 1
    )

    return model, trained_model


def create_model_layers(model_parameters):
    session["temp_model_name"] = model_parameters["name"]

    model_layers = {
        "input_layer": [
            int(model_parameters["input_layer_filter_size"]),
            int(model_parameters["input_layer_kernel_size"]),
            model_parameters["input_layer_activation"],
        ]
    }

    model_layers["output_layer"] = [ "Dense", 6, model_parameters["output_layer_activation"] ]

    number_of_layers_before_flatten = int(model_parameters["number_of_layers_before_flatten"])
    number_of_layers_after_flatten = int(model_parameters["number_of_layers_after_flatten"])

    counter = 0
    layer_number = 0

    model_layers["before_flatten"] = dict()
    model_layers["after_flatten"] = dict()
    
    for parameter in model_parameters:
        if counter < 6:
            counter += 1
            continue

        else:
            parameter_name = parameter.split('_')

            if parameter_name[0] != "output" and parameter_name[-1] == "class":
                layer_number += 1

                if number_of_layers_before_flatten > 0:
                    model_layers["before_flatten"][f"layer_{layer_number}"] = [model_parameters[parameter]]

                    before_flatten = True

                    number_of_layers_before_flatten -= 1

                elif number_of_layers_after_flatten > 0:
                    model_layers["after_flatten"][f"layer_{layer_number}"] = [model_parameters[parameter]]

                    before_flatten = False

                    number_of_layers_after_flatten -= 1

            elif parameter_name[0] == "output":
                break

            else:
                if before_flatten:
                    model_layers["before_flatten"][f"layer_{layer_number}"].append(model_parameters[parameter])

                else:
                    model_layers["after_flatten"][f"layer_{layer_number}"].append(model_parameters[parameter])

    return model_layers

def create_model(model_layers):
    model = Models.Sequential()

    model.add(
        Layers.Conv2D(
            model_layers["input_layer"][0],
            (model_layers["input_layer"][1], model_layers["input_layer"][1]),
            activation = model_layers["input_layer"][2],
            padding = "same",
            input_shape = ( 150, 150, 3 )
        )
    )

    for layer in model_layers["before_flatten"]:
        if model_layers["before_flatten"][layer][0] == "Conv2D":
            model.add(
                Layers.Conv2D(
                    filters = int(model_layers["before_flatten"][layer][1]),
                    kernel_size = ( int(model_layers["before_flatten"][layer][2]), int(model_layers["before_flatten"][layer][2]) ),
                    activation = model_layers["before_flatten"][layer][3],
                     padding = "same"
                )
            )

        elif model_layers["before_flatten"][layer][0] == "MaxPool2D":
            model.add(
                Layers.MaxPool2D(
                    ( int(model_layers["before_flatten"][layer][1]), int(model_layers["before_flatten"][layer][1]) )
                )
            )

        elif model_layers["before_flatten"][layer][0] == "Dropout":
            model.add(
                Layers.Dropout(
                    rate = float(model_layers["before_flatten"][layer][1])
                )
            )

        elif model_layers["before_flatten"][layer][0] == "Dense":
            model.add(
                Layers.Dense(
                    int(model_layers["before_flatten"][layer][1]),
                    activation = model_layers["before_flatten"][layer][2]
                )
            )

    model.add( Layers.Flatten() )

    for layer in model_layers["after_flatten"]:
        if model_layers["after_flatten"][layer][0] == "Dropout":
            model.add(
                Layers.Dropout(
                    rate = float(model_layers["after_flatten"][layer][1])
                )
            )

        elif model_layers["after_flatten"][layer][0] == "Dense":
            model.add(
                Layers.Dense(
                    int(model_layers["after_flatten"][layer][1]),
                    activation = model_layers["after_flatten"][layer][2]
                )
            )

    model.add(
        Layers.Dense(
            6,
            activation = model_layers["output_layer"][2]
        )
    )

    return model

def get_training_parameters(model_parameters):
    training_parameters = {
        "number_of_epochs": 3,
        "batch_size": 128,
        "multi_processing": True if model_parameters["multi_processing"] == "enabled" else False,
        "learning_rate": float(model_parameters["learning_rate"]),
        "optimization_function": model_parameters["optimization_function"],
        "loss_function": model_parameters["loss_function"],
    }

    return training_parameters

def get_optimizer(optimization_function, learning_rate):
    if optimization_function == "Adam":
        optimization_function = Optimizer.Adam(learning_rate = learning_rate)

    elif optimization_function == "SGD":
        optimization_function = Optimizer.SGD(learning_rate = learning_rate)

    elif optimization_function == "RMSprop":
        optimization_function = Optimizer.RMSprop(learning_rate = learning_rate)

    elif optimization_function == "Adagrad":
        optimization_function = Optimizer.Adagrad(learning_rate = learning_rate)

    elif optimization_function == "Adadelta":
        optimization_function = Optimizer.Adadelta(learning_rate = learning_rate)

    elif optimization_function == "Adamax":
        optimization_function = Optimizer.Adamax(learning_rate = learning_rate)

    elif optimization_function == "Nadam":
        optimization_function = Optimizer.Nadam(learning_rate = learning_rate)

    return optimization_function

def save_model_local(model):
    model_id = uuid4().hex
    model_path = f"static/models/temp/{session['id']}/{model_id}"

    create_path_if_not_exists(f"static/models/temp/{session['id']}")
    create_path_if_not_exists(model_path)

    model.save(f"{model_path}/model.h5")

    session["temp_model_id"] = model_id

def save_model_accuracy_local(model):
    plot.plot(model.history["accuracy"])
    plot.plot(model.history["val_accuracy"])
    plot.ylabel("Accuracy")
    plot.xlabel("Epoch")
    plot.legend(["Training Accuracy", "Validation Accuracy"], loc = "upper left")

    plot.savefig(f"static/models/temp/{session['id']}/{session['temp_model_id']}/model_accuracy.png")

def save_model_loss_local(model):
    plot.plot(model.history['loss'])
    plot.plot(model.history['val_loss'])
    plot.ylabel('Loss')
    plot.xlabel('Epoch')
    plot.legend(['Training Loss', 'Validation Loss'], loc='upper left')

    plot.savefig(f"static/models/temp/{session['id']}/{session['temp_model_id']}/model_loss.png")

def save_model_to_cloud():
    create_model_document()

    upload_model_to_cloud()


def create_model_document():
    database.collection("users").document(session["user"]["id"]).collection("models").document(session["temp_model_id"]).set({
        "name": session["temp_model_name"]
    })

def upload_model_to_cloud():
    cloud_model_path = f"{session['user']['id']}/models/{session['temp_model_id']}"
    local_model_path = f"static/models/temp/{session['id']}/{session['temp_model_id']}"

    model_blob = cloud_storage.blob(f"{cloud_model_path}/model.h5")
    model_blob.upload_from_filename(f"{local_model_path}/model.h5")

    model_accuracy_blob = cloud_storage.blob(f"{cloud_model_path}/model_accuracy.png")
    model_accuracy_blob.upload_from_filename(f"{local_model_path}/model_accuracy.png")

    model_loss_blob = cloud_storage.blob(f"{cloud_model_path}/model_loss.png")
    model_loss_blob.upload_from_filename(f"{local_model_path}/model_loss.png")


def finish_classification():
    clear_uploaded_images()

    clear_model_files()

    clear_session()

def clear_uploaded_images():
    images_path = f"{UPLOADS_FOLDER}/temp/{session['id']}"
    
    rmtree(images_path, ignore_errors = True) if path.exists(images_path) else False
    
def clear_model_files():
    
    rmtree(session["model_path"], ignore_errors = True) if path.exists(session["model_path"]) else False

def clear_session():
    if "temp_model_id" in session:
        del session["temp_model_id"]

    if "temp_model_name" in session:
        del session["temp_model_name"]

    if "model_path" in session:
        del session["model_path"]
        
    if "predictions" in session:
        del session["predictions"]

    if "number_of_images" in session:
        del session["number_of_images"]

    if "number_of_corrections" in session:
        del session["number_of_corrections"]
    
    if "model_selected" in session:
        del session["model_selected"]

    if "images_uploaded" in session:
        del session["images_uploaded"]
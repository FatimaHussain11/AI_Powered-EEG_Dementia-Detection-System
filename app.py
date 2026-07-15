"""
app.py
==================================================================
Flask backend for the AI-Powered EEG Dementia Detection System.
==================================================================
"""

import os
import logging
import traceback

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

import predict
import utils

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
MODELS_FOLDER = os.path.join(BASE_DIR, "models")

MAX_CONTENT_LENGTH_MB = 50

MODEL_BINARY = "binary"
MODEL_THREE_CLASS = "three-class"
VALID_MODEL_VALUES = {MODEL_BINARY, MODEL_THREE_CLASS}

app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MODELS_FOLDER"] = MODELS_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH_MB * 1024 * 1024

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def ensure_required_folders():
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    os.makedirs(app.config["MODELS_FOLDER"], exist_ok=True)


ensure_required_folders()
predict.load_all_models()


def run_model_inference(model_choice, file_path):
    if model_choice == MODEL_BINARY:
        return predict.run_binary_prediction(file_path)
    if model_choice == MODEL_THREE_CLASS:
        return predict.run_three_class_prediction(file_path)
    raise ValueError(f"Unrecognized model selection: '{model_choice}'")


def build_success_response(result, model_choice):
    """
    Constructs the exact long string names and short probability dictionary keys
    required by your original script.js layout engine.
    """
    raw_key = result["prediction"]  # This will be 'AD', 'HC', or 'FTD'

    # Translate target outputs to display text names for the summary header label
    mapping = {
        "HC": "Healthy Control",
        "AD": "Alzheimer's Disease",
        "FTD": "Frontotemporal Dementia"
    }

    predicted_display_name = mapping.get(raw_key, raw_key)
    description = predict.get_description_for_prediction(raw_key)
    model_label = "Binary Classification" if model_choice == MODEL_BINARY else "Three-Class Classification"

    return {
        "prediction": predicted_display_name,  # "Healthy Control"
        "confidence": result["confidence"],
        "model": model_label,
        "probabilities": result["probabilities"],  # Keep short keys {"HC": 66.7, "AD": 33.3}
        "description": description,
        "status": "Success",
    }


def build_error_response(message, status_code):
    return (
        jsonify({
            "prediction": None,
            "confidence": None,
            "model": None,
            "probabilities": {},
            "description": "",
            "status": "Error",
            "message": message,
        }),
        status_code,
    )


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict_route():
    try:
        if "file" not in request.files:
            return build_error_response("No file was included in the request.", 400)

        uploaded_file = request.files["file"]

        if uploaded_file.filename == "":
            return build_error_response("No file was selected.", 400)

        if not utils.allowed_file(uploaded_file.filename):
            return build_error_response(
                f"Unsupported file type. Allowed format(s): "
                f"{', '.join(sorted(utils.ALLOWED_EXTENSIONS))}.",
                400,
            )

        model_choice = request.form.get("model", "").strip().lower()

        if not model_choice:
            return build_error_response("No prediction model was selected.", 400)

        if model_choice not in VALID_MODEL_VALUES:
            return build_error_response(
                f"Invalid model '{model_choice}'. Expected 'binary' or 'three-class'.",
                400,
            )

        try:
            file_path = utils.save_uploaded_file(uploaded_file, app.config["UPLOAD_FOLDER"])
        except Exception as exc:
            logger.error("Failed to save uploaded file: %s", exc)
            return build_error_response("Failed to save the uploaded file on the server.", 500)

        logger.info("Saved uploaded EEG file to %s", file_path)

        try:
            result = run_model_inference(model_choice, file_path)
        except predict.ModelNotLoadedError as exc:
            logger.warning("Model not loaded: %s", exc)
            return build_error_response(str(exc), 503)
        except predict.EEGPreprocessingError as exc:
            logger.warning("Preprocessing failed: %s", exc)
            return build_error_response(str(exc), 422)
        except ValueError as exc:
            logger.warning("Invalid model selection: %s", exc)
            return build_error_response(str(exc), 400)

        response_payload = build_success_response(result, model_choice)
        return jsonify(response_payload), 200

    except Exception as exc:
        logger.error("Unexpected error in /predict: %s\n%s", exc, traceback.format_exc())
        return build_error_response(
            "An unexpected error occurred while processing the EEG file.", 500
        )


@app.errorhandler(413)
def handle_file_too_large(_error):
    return build_error_response(
        f"The uploaded file is too large. Maximum size is {MAX_CONTENT_LENGTH_MB}MB.",
        413,
    )


@app.errorhandler(404)
def handle_not_found(_error):
    return build_error_response("The requested resource was not found.", 404)


@app.errorhandler(500)
def handle_internal_error(_error):
    return build_error_response("An internal server error occurred.", 500)


if __name__ == "__main__":
    # Debug mode is now controlled by an environment variable and
    # defaults to OFF. Never run with debug=True on a public server —
    # it exposes an interactive code-execution console on error pages.
    debug_mode = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=5000, debug=debug_mode)
"""
utils.py
==================================================================
Small shared helpers used by app.py. Kept separate from predict.py
so file-handling/validation logic doesn't get tangled up with the
machine-learning pipeline.
==================================================================
"""

import os
from datetime import datetime, timezone
from werkzeug.utils import secure_filename

# The trained model (see predict.py) expects EEGLAB .set recordings.
# Only add more extensions here once a matching preprocessing path
# exists in predict.py for that format.
# Updated to support all frontend formats matching app.py configuration
ALLOWED_EXTENSIONS = {"csv", "edf", "mat", "txt", "npy", "set"}

def allowed_file(filename):
    """Checks whether the uploaded file has a supported extension."""
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def save_uploaded_file(uploaded_file, upload_folder):
    """
    Saves the uploaded EEG file to disk using a safe, timestamped
    filename to avoid collisions and path-traversal issues.

    Note: EEGLAB .set files often reference a companion .fdt file
    (for the raw data) that must sit next to the .set file with a
    matching base name. If your export uses .fdt sidecar files,
    make sure the frontend/backend also uploads and saves that file
    alongside the .set file using the same base name.

    Returns:
        str: the full path to the saved .set file on disk.
    """
    original_name = secure_filename(uploaded_file.filename)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    safe_filename = f"{timestamp}_{original_name}"

    destination_path = os.path.join(upload_folder, safe_filename)
    uploaded_file.save(destination_path)
    return destination_path

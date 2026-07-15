"""
predict.py
==================================================================
The real EEG dementia detection pipeline, loading production models.
==================================================================
"""

import os
import logging
import joblib
import numpy as np
import mne

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# NOTE: the binary model and the three-class model were trained on two
# DIFFERENT feature sets (see comments in extract_mock_features_binary
# and extract_band_power_features below). They are NOT interchangeable,
# so each pipeline has its own artifact paths and its own feature
# extraction function.
ARTIFACT_PATHS = {
    "binary": {
        "model": os.path.join(MODELS_DIR, "production_eeg_model.pkl"),
        "scaler": os.path.join(MODELS_DIR, "production_scaler.pkl"),
    },
    "three_class": {
        "model": os.path.join(MODELS_DIR, "production_eeg_model_3class.pkl"),
        "scaler": os.path.join(MODELS_DIR, "production_scaler_3class.pkl"),
        "selector": os.path.join(MODELS_DIR, "production_selector_3class.pkl"),
        "label_encoder": os.path.join(MODELS_DIR, "production_label_encoder_3class.pkl"),
    },
}

CLASS_DISPLAY_NAMES = {
    "AD": "Alzheimer's Disease",
    "FTD": "Frontotemporal Dementia",
    "HC": "Healthy Control",
}

CLASS_DESCRIPTIONS = {
    "AD": "Alzheimer's disease is a progressive neurodegenerative disorder that gradually impairs memory, reasoning, and daily functioning.",
    "FTD": "Frontotemporal dementia primarily affects the frontal and temporal lobes, impacting behavior, personality, and language.",
    "HC": "No dementia-related pattern was detected in the analyzed EEG signal.",
}

BANDS = {"delta": (0.5, 4), "theta": (4, 8), "alpha": (8, 12), "beta": (12, 30)}
CHANNELS = [
    "Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
    "F7", "F8", "T3", "T4", "T5", "T6", "Fz", "Cz", "Pz",
]


class ModelNotLoadedError(Exception):
    """Raised when a prediction is requested before its artifacts are loaded."""


class EEGPreprocessingError(Exception):
    """Raised when an uploaded file can't be read or featurized."""


_artifacts = {
    "binary": None,
    "three_class": None,
}


def load_all_models():
    """Loads saved pipeline artifacts into memory once at application startup."""
    for pipeline_key, paths in ARTIFACT_PATHS.items():
        try:
            _artifacts[pipeline_key] = _load_pipeline_artifacts(paths)
            logger.info("Loaded '%s' pipeline artifacts successfully.", pipeline_key)
        except FileNotFoundError as exc:
            logger.warning("'%s' pipeline artifacts not found yet: %s", pipeline_key, exc)
            _artifacts[pipeline_key] = None
        except Exception as exc:
            logger.error("Failed to load '%s' pipeline artifacts: %s", pipeline_key, exc)
            _artifacts[pipeline_key] = None


def _load_pipeline_artifacts(paths):
    for label, path in paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing '{label}' artifact at {path}")
    loaded = {}
    for label, path in paths.items():
        loaded[label] = joblib.load(path)
    return loaded


def is_pipeline_ready(pipeline_key):
    return _artifacts.get(pipeline_key) is not None


# ------------------------------------------------------------------
# BINARY pipeline features (placeholder — real graph-signal-processing
# feature extraction code was not present in the training notebook,
# so this still returns mock data. Do NOT trust binary predictions
# until this is replaced with the real feature pipeline.)
# ------------------------------------------------------------------
def extract_mock_features_binary(set_file_path):
    try:
        mock_features = np.zeros(36)
        mock_features[2] = 0.5
        mock_features[5] = 300.0
        mock_features[6] = 1.4
        return mock_features
    except Exception as exc:
        raise EEGPreprocessingError(f"Failed to read feature structures from data: {exc}")


# ------------------------------------------------------------------
# THREE-CLASS pipeline features (real — ported directly from the
# training notebook's predict_patient_dementia() function).
# ------------------------------------------------------------------
def extract_band_power_features(set_file_path):
    if not os.path.exists(set_file_path):
        raise EEGPreprocessingError(f"File path not found: {set_file_path}")

    try:
        raw = mne.io.read_raw_eeglab(set_file_path, preload=True, verbose=False)
        existing_channels = [ch for ch in CHANNELS if ch in raw.ch_names]
        raw.pick(existing_channels)
        raw.filter(l_freq=0.5, h_freq=30.0, fir_design="firwin", verbose=False)

        epochs = mne.make_fixed_length_epochs(raw, duration=4.0, preload=True, verbose=False)
        psd_spectrum = epochs.compute_psd(method="welch", fmin=0.5, fmax=30.0, verbose=False)
        psds, freqs = psd_spectrum.get_data(return_freqs=True)

        X_epochs = []
        for ep in range(psds.shape[0]):
            feature_vector = []
            for ch in range(psds.shape[1]):
                powers = {}
                for band_name, (fmin, fmax) in BANDS.items():
                    idx = np.where((freqs >= fmin) & (freqs <= fmax))[0]
                    powers[band_name] = np.mean(psds[ep, ch, idx])

                slowing_ratio = (powers["delta"] + powers["theta"]) / (
                    powers["alpha"] + powers["beta"] + 1e-10
                )
                feature_vector.append(slowing_ratio)

                total_power = sum(powers.values()) + 1e-10
                for band_name in BANDS.keys():
                    feature_vector.append(powers[band_name] / total_power)

            X_epochs.append(feature_vector)

        return np.array(X_epochs)  # shape: (n_epochs, 95)

    except EEGPreprocessingError:
        raise
    except Exception as exc:
        raise EEGPreprocessingError(
            f"Failed to read/featurize EEG file (expects an EEGLAB .set file): {exc}"
        )


def predict_three_class(set_file_path, model, scaler, selector, label_encoder):
    """
    Real 3-class inference: extracts band-power features per epoch,
    scales + selects them through the production pipeline, then
    soft-votes epoch-level probabilities into one patient-level result.
    """
    X_epochs = extract_band_power_features(set_file_path)

    X_scaled = scaler.transform(X_epochs)
    X_selected = selector.transform(X_scaled)

    epoch_probabilities = model.predict_proba(X_selected)
    patient_probabilities = np.mean(epoch_probabilities, axis=0)

    probabilities = {
        label_encoder.classes_[i]: round(float(patient_probabilities[i]) * 100, 1)
        for i in range(len(label_encoder.classes_))
    }
    predicted_class_key = max(probabilities, key=probabilities.get)

    return {
        "prediction": predicted_class_key,
        "confidence": probabilities[predicted_class_key],
        "probabilities": probabilities,
    }


def predict_binary_mock(set_file_path, model, scaler):
    """
    Placeholder binary inference — still using mock features.
    See extract_mock_features_binary() docstring above.
    """
    features = extract_mock_features_binary(set_file_path)
    scaled_features = scaler.transform([features])
    probability_array = model.predict_proba(scaled_features)[0]

    probabilities = {
        "AD": round(float(probability_array[0]) * 100, 1),
        "HC": round(float(probability_array[1]) * 100, 1),
    }
    predicted_class_key = max(probabilities, key=probabilities.get)

    return {
        "prediction": predicted_class_key,
        "confidence": probabilities[predicted_class_key],
        "probabilities": probabilities,
    }


def run_binary_prediction(set_file_path):
    artifacts = _artifacts.get("binary")
    if artifacts is None:
        raise ModelNotLoadedError("The binary pipeline artifacts are not fully initialized.")
    return predict_binary_mock(set_file_path, model=artifacts["model"], scaler=artifacts["scaler"])


def run_three_class_prediction(set_file_path):
    artifacts = _artifacts.get("three_class")
    if artifacts is None:
        raise ModelNotLoadedError("The three-class pipeline artifacts are not fully initialized.")
    return predict_three_class(
        set_file_path,
        model=artifacts["model"],
        scaler=artifacts["scaler"],
        selector=artifacts["selector"],
        label_encoder=artifacts["label_encoder"],
    )


def get_description_for_prediction(predicted_class_key):
    clean_key = "AD" if "AD" in predicted_class_key or "Alzheimer" in predicted_class_key else predicted_class_key
    clean_key = "HC" if "HC" in clean_key or "Healthy" in clean_key else clean_key
    clean_key = "FTD" if "FTD" in clean_key or "Frontotemporal" in clean_key else clean_key

    return CLASS_DESCRIPTIONS.get(clean_key, "No description is available for this prediction.")

import json
import logging
import tempfile
from pathlib import Path

import librosa
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

from flask import Flask, jsonify, render_template, request
from flask_sock import Sock
from tensorflow.keras.models import load_model


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

SAMPLE_RATE = 16000
WINDOW_SECONDS = 6
HOP_SECONDS = 2

WINDOW_SAMPLES = SAMPLE_RATE * WINDOW_SECONDS
HOP_SAMPLES = SAMPLE_RATE * HOP_SECONDS

DISTRESS_THRESHOLD = 0.5

YAMNET_PATH = BASE_DIR / "models" / "yamnet"
CLASSIFIER_PATH = BASE_DIR / "emergency_audio_classifier.keras"

MAX_UPLOAD_BYTES = 50 * 1024 * 1024
MAX_AUDIO_SECONDS = 600  # Maximum uploaded recording: 10 minutes

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("WASP")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES

sock = Sock(app)

yamnet = None
classifier = None


# ============================================================
# MODEL LOADING
# ============================================================

def load_models():
    global yamnet, classifier

    if not YAMNET_PATH.exists():
        raise FileNotFoundError(
            f"Local YAMNet model not found: {YAMNET_PATH}"
        )

    if not CLASSIFIER_PATH.exists():
        raise FileNotFoundError(
            f"Classifier not found: {CLASSIFIER_PATH}"
        )

    logger.info("Loading local YAMNet...")
    yamnet = hub.load(str(YAMNET_PATH))

    logger.info("Loading emergency classifier...")
    classifier = load_model(str(CLASSIFIER_PATH))

    logger.info("WASP models loaded successfully.")


# ============================================================
# AUDIO CLASSIFICATION
# ============================================================

def classify_window(audio_window):
    """Classify exactly six seconds of 16 kHz mono float audio."""

    audio_window = np.asarray(
        audio_window, dtype=np.float32
    ).reshape(-1)

    if len(audio_window) != WINDOW_SAMPLES:
        raise ValueError("Expected a six-second audio window.")

    _, embeddings, _ = yamnet(audio_window)

    # Must match the embedding aggregation used during training.
    mean_embedding = tf.reduce_mean(
        embeddings, axis=0
    ).numpy().astype(np.float32)

    prediction = classifier.predict(
        mean_embedding.reshape(1, -1),
        verbose=0,
    )

    probability = float(
        np.asarray(prediction).reshape(-1)[0]
    )

    if not np.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("Invalid classifier probability.")

    return {
        "label": (
            "distress"
            if probability >= DISTRESS_THRESHOLD
            else "normal"
        ),
        "probability": round(probability, 4),
        "threshold": DISTRESS_THRESHOLD,
    }


def classify_recording(audio):
    """
    Classify a recording using six-second windows
    with a two-second hop.

    Short recordings are zero-padded to six seconds.
    The final window is aligned to the recording end
    when necessary, so the tail is not ignored.
    """

    audio = np.asarray(audio, dtype=np.float32).reshape(-1)

    if audio.size == 0:
        raise ValueError("The uploaded audio contains no samples.")

    if len(audio) > SAMPLE_RATE * MAX_AUDIO_SECONDS:
        raise ValueError("Audio exceeds the ten-minute limit.")

    # Pad clips shorter than one complete window.
    if len(audio) < WINDOW_SAMPLES:
        padded = np.pad(
            audio,
            (0, WINDOW_SAMPLES - len(audio)),
            mode="constant",
        )

        result = classify_window(padded)
        result.update({
            "start_seconds": 0.0,
            "end_seconds": round(len(audio) / SAMPLE_RATE, 2),
            "padded": True,
        })

        return [result]

    # Regular overlapping windows.
    starts = list(
        range(
            0,
            len(audio) - WINDOW_SAMPLES + 1,
            HOP_SAMPLES,
        )
    )

    # Add a final end-aligned window if the recording has
    # a tail that the regular windows would otherwise miss.
    final_start = len(audio) - WINDOW_SAMPLES

    if starts[-1] != final_start:
        starts.append(final_start)

    results = []

    for start in starts:
        end = start + WINDOW_SAMPLES
        result = classify_window(audio[start:end])

        result.update({
            "start_seconds": round(start / SAMPLE_RATE, 2),
            "end_seconds": round(
                min(end, len(audio)) / SAMPLE_RATE, 2
            ),
            "padded": False,
        })

        results.append(result)

    return results


# ============================================================
# WEBPAGE AND HEALTH
# ============================================================

@app.get("/")
def index():
    return render_template("index.html")


@app.get("/health")
def health():
    ready = yamnet is not None and classifier is not None

    return jsonify({
        "status": "ready" if ready else "not_ready",
        "yamnet_loaded": yamnet is not None,
        "classifier_loaded": classifier is not None,
        "inference": "local",
        "window_seconds": WINDOW_SECONDS,
        "hop_seconds": HOP_SECONDS,
    }), 200 if ready else 503


# ============================================================
# UPLOADED AUDIO CLASSIFICATION
# ============================================================

@app.post("/classify")
def classify_upload():
    if classifier is None or yamnet is None:
        return jsonify({"error": "Models are not loaded."}), 503

    if "audio" not in request.files:
        return jsonify({
            "error": "Choose an audio file first."
        }), 400

    uploaded = request.files["audio"]

    if not uploaded.filename:
        return jsonify({"error": "Empty filename."}), 400

    suffix = Path(uploaded.filename).suffix.lower()

    allowed_extensions = {
        ".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".webm"
    }

    if suffix not in allowed_extensions:
        return jsonify({
            "error": "Unsupported audio format."
        }), 400

    temp_path = None

    try:
        # The temporary file is deleted after processing.
        with tempfile.NamedTemporaryFile(
            suffix=suffix, delete=False
        ) as temp:
            temp_path = Path(temp.name)
            uploaded.save(temp_path)

        # Decode, convert to mono, and resample to 16 kHz.
        audio, _ = librosa.load(
            str(temp_path),
            sr=SAMPLE_RATE,
            mono=True,
        )

        if len(audio) == 0:
            return jsonify({"error": "No audio samples found."}), 400

        if len(audio) > SAMPLE_RATE * MAX_AUDIO_SECONDS:
            return jsonify({
                "error": "Maximum recording length is 10 minutes."
            }), 400

        predictions = classify_recording(audio)

        distress_count = sum(
            item["label"] == "distress"
            for item in predictions
        )

        return jsonify({
            "filename": uploaded.filename,
            "duration_seconds": round(
                len(audio) / SAMPLE_RATE, 2
            ),
            "window_seconds": WINDOW_SECONDS,
            "hop_seconds": HOP_SECONDS,
            "total_windows": len(predictions),
            "distress_windows": distress_count,
            "overall_label": (
                "distress" if distress_count > 0 else "normal"
            ),
            "predictions": predictions,
        })

    except Exception:
        logger.exception("Uploaded audio classification failed.")
        return jsonify({
            "error": (
                "Could not classify this audio. "
                "Check the file format and server logs."
            )
        }), 400

    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


# ============================================================
# LIVE WEBSOCKET CLASSIFICATION
# ============================================================

@sock.route("/live")
def live_audio(ws):
    if yamnet is None or classifier is None:
        ws.send(json.dumps({"error": "Models are not loaded."}))
        return

    # Each WebSocket connection has its own buffer.
    buffer = np.empty(0, dtype=np.float32)

    total_samples = 0
    samples_since_prediction = 0

    logger.info("Live audio client connected.")

    try:
        while True:
            message = ws.receive()

            if message is None:
                break

            if not isinstance(message, bytes):
                ws.send(json.dumps({
                    "error": "Send binary PCM audio."
                }))
                continue

            # 10 seconds maximum per chunk.
            if len(message) > SAMPLE_RATE * 10 * 2:
                ws.send(json.dumps({"error": "Chunk too large."}))
                continue

            if len(message) == 0:
                continue

            if len(message) % 2:
                ws.send(json.dumps({"error": "Invalid PCM chunk."}))
                continue

            pcm = np.frombuffer(message, dtype="<i2")
            chunk = pcm.astype(np.float32) / 32768.0

            total_samples += len(chunk)
            samples_since_prediction += len(chunk)

            buffer = np.concatenate((buffer, chunk))

            if len(buffer) > WINDOW_SAMPLES:
                buffer = buffer[-WINDOW_SAMPLES:]

            if len(buffer) < WINDOW_SAMPLES:
                continue

            if samples_since_prediction < HOP_SAMPLES:
                continue

            try:
                result = classify_window(buffer)

                result.update({
                    "window_seconds": WINDOW_SECONDS,
                    "hop_seconds": HOP_SECONDS,
                    "audio_end_seconds": round(
                        total_samples / SAMPLE_RATE, 2
                    ),
                })

                ws.send(json.dumps(result))

            except Exception:
                logger.exception("Live inference failed.")
                ws.send(json.dumps({
                    "error": "Could not classify this window."
                }))

            finally:
                samples_since_prediction %= HOP_SAMPLES

    except Exception:
        logger.exception("WebSocket connection ended.")

    finally:
        logger.info("Live audio client disconnected.")


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":
    load_models()

    # Local development server. Restrict network access
    # before using this on an organisation's LAN.
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
        threaded=True,
    )

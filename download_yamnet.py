
from pathlib import Path
import tensorflow as tf
import tensorflow_hub as hub

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models" / "yamnet"

MODEL_DIR.parent.mkdir(parents=True, exist_ok=True)

print("Downloading YAMNet from TensorFlow Hub...")
yamnet = hub.load("https://tfhub.dev/google/yamnet/1")

print(f"Saving YAMNet to: {MODEL_DIR}")
tf.saved_model.save(yamnet, str(MODEL_DIR))

print("YAMNet saved successfully.")
print("Verify the model works locally before disconnecting from the internet.")

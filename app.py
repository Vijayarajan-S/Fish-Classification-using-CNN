from flask import Flask, render_template, request, jsonify
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename

# -------------------- CONFIG --------------------
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load your trained CNN model
MODEL_PATH = r"C:\Users\vijay\Desktop\FishClassification\models\best_resnet_model.keras"
if os.path.exists(MODEL_PATH):
    model = load_model(MODEL_PATH)
    print("✅ Model loaded successfully!")
else:
    raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")

# Define the class labels
CLASS_LABELS = [
    'animal fish', 'animal fish bass', 'fish sea_food black_sea_sprat',
    'fish sea_food gilt_head_bream', 'fish sea_food hourse_mackerel',
    'fish sea_food red_mullet', 'fish sea_food red_sea_bream', 
    'fish sea_food sea_bass', 'fish sea_food shrimp',
    'fish sea_food striped_red_mullet', 'fish sea_food trout'
]

# -------------------- ROUTES --------------------
@app.route('/')
def home():
    return render_template("index.html")

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({"status": "error", "message": "No file part"}), 400

    img_file = request.files['image']
    if img_file.filename == '':
        return jsonify({"status": "error", "message": "No selected file"}), 400

    # Secure filename and save
    filename = secure_filename(img_file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    img_file.save(file_path)

    # -------------------- IMAGE PREPROCESSING --------------------
    img = image.load_img(file_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize

    # -------------------- PREDICTION --------------------
    predictions = model.predict(img_array)

    # Ensure predictions is a tensor
    predictions = tf.convert_to_tensor(predictions)

    # Get predicted class index and confidence
    predicted_class_idx = tf.argmax(predictions, axis=-1).numpy()[0]
    confidence = tf.reduce_max(predictions).numpy()

    result = {
        "status": "success",
        "predicted_class": CLASS_LABELS[predicted_class_idx],
        "confidence": round(float(confidence) * 100, 2)
    }

    return jsonify(result)

# -------------------- MAIN --------------------
if __name__ == "__main__":
    app.run(debug=True)

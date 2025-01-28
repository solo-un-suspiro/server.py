from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
from PIL import Image
import io
import cv2
from werkzeug.utils import secure_filename
import os
import traceback

app = Flask(__name__)
CORS(app)

# Cargar el modelo entrenado
try:
    model = joblib.load('star_model.pkl')
except FileNotFoundError:
    print("Error: No se pudo encontrar el archivo 'star_model.pkl'")
    model = None

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def detect_stars(img):
    # Implementa aquí la lógica de detección de estrellas
    # Este es un ejemplo simplificado, deberías reemplazarlo con tu lógica real
    
    height, width = img.shape[:2]
    stars = {
        'fourPoint': [{'x': 0.2, 'y': 0.3, 'radius': 0.05}],
        'fivePoint': [{'x': 0.5, 'y': 0.5, 'radius': 0.07}],
        'sixPoint': [{'x': 0.8, 'y': 0.7, 'radius': 0.06}]
    }
    return stars

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and allowed_file(file.filename):
        try:
            filename = secure_filename(file.filename)
            img = Image.open(file.stream)
            img = np.array(img)

            # Preprocesar la imagen
            img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            
            # Redimensionar la imagen a 64x64 (ya que 64*64 = 4096)
            img_resized = cv2.resize(img_gray, (64, 64))

            # Realizar la predicción
            if model is None:
                return jsonify({'error': 'Model not loaded'}), 500
            prediction = model.predict([img_resized.flatten()])

            # Detectar estrellas
            stars = detect_stars(img)

            return jsonify({
                'message': f"Predicción: {prediction[0]}",
                'stars': stars
            })
        except Exception as e:
            print(f"Error processing image: {str(e)}")
            print(traceback.format_exc())
            return jsonify({'error': str(e)}), 500
    else:
        return jsonify({'error': 'File type not allowed'}), 400

if __name__ == '__main__':
    # Configuración para Render: obtener el puerto desde la variable de entorno
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)

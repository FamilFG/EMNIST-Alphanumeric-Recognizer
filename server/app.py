from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from PIL import Image
import io
import base64
from predict import Predictor

app = Flask(__name__)
CORS(app)

model = tf.keras.models.load_model('../model/emnist_cnn_new.h5')
predictor = Predictor(model)
print("Model loaded successfully!")

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        image_data = data.get('image')
        
        if not image_data:
            return jsonify({"error": "No image provided"}), 400
        
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        img = Image.open(io.BytesIO(image_bytes))
        
        results = predictor.predict(img, top_k=5)
        
        return jsonify({
            "success": True,
            "prediction": results[0]["character"],
            "confidence": results[0]["confidence"],
            "top5": results
        })
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
import numpy as np
from utils import get_char, preprocess_image

class Predictor:
    def __init__(self, model):
        self.model = model
    
    def predict(self, img, top_k=5):
        processed = preprocess_image(img)
        
        predictions = self.model.predict(processed, verbose=0)[0]
        
        top_indices = np.argsort(predictions)[-top_k:][::-1]
        results = []
        
        for idx in top_indices:
            results.append({
                "character": get_char(idx),
                "confidence": float(predictions[idx] * 100)
            })
        
        return results
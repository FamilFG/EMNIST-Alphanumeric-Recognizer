import numpy as np
from PIL import Image

def get_char(label):
    if label < 10: 
        return str(label)
    elif label < 36: 
        return chr(ord('A') + label - 10)
    else: 
        return chr(ord('a') + label - 36)

def preprocess_image(img):
    img = img.convert('L')
    
    img = img.transpose(Image.FLIP_LEFT_RIGHT)
    
    img_array = np.array(img)
    
    if np.mean(img_array) > 127:
        img_array = 255 - img_array
    
    threshold = 30
    rows = np.any(img_array > threshold, axis=1)
    cols = np.any(img_array > threshold, axis=0)
    
    if rows.any() and cols.any():
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        img_array = img_array[rmin:rmax+1, cmin:cmax+1]
    
    h, w = img_array.shape
    size = max(h, w)
    square = np.zeros((size, size), dtype=np.uint8)
    y_offset = (size - h) // 2
    x_offset = (size - w) // 2
    square[y_offset:y_offset+h, x_offset:x_offset+w] = img_array
    
    border = int(size * 0.2)
    square = np.pad(square, border, mode='constant', constant_values=0)
    
    img = Image.fromarray(square)
    img = img.resize((28, 28), Image.Resampling.LANCZOS)
    img_array = np.array(img)
    
    return (img_array / 255.0).reshape(1, 28, 28, 1)
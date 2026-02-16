# EMNIST Alphanumeric Recognizer

Simple CNN-based EMNIST alphanumeric classifier with a small web client and a Python server. Use the web page to draw a character and send it to the API for prediction.

## Structure
- client/: current static frontend (planned React replacement)
- server/: prediction API
- model/: trained model file
- emnist/: dataset files

## Quick start
1) Install server dependencies:
   - pip install -r server/requirements.txt
2) Start the server:
   - cd server
   - python app.py
3) Open the client:
   - open client/index.html in a browser

## React client (planned)
- A React-based frontend is planned for the client/ folder.
- This README will be updated with React dev server steps once added.

## API
- Default server runs on http://127.0.0.1:5000/.
- The client sends a drawn image to the prediction endpoint exposed in server/app.py.

## Training (optional)
- Use train_emnist.py to retrain the CNN on the EMNIST ByClass dataset.
- The dataset files are already under emnist/byclass/.
- After training, replace model/emnist_cnn_new.h5 or update the path in server/predict.py.

## Troubleshooting
- If the server exits on start, verify Python version and reinstall dependencies.
- If predictions look wrong, confirm the model path and that the image preprocessing in server/utils.py matches training.

## Notes
- The model file is in model/emnist_cnn_new.h5.
- If you retrain, update the model path in server/predict.py.

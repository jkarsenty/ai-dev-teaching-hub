from flask import Flask, jsonify, request
from model import load_model, predict

app = Flask(__name__)

# model loading
model = load_model()

@app.route('/')
def home():
    return "Welcome to the Alyra Deployment API!"

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "message": "API is running smoothly."})

@app.route('/predict', methods=['POST'])
def predict_sentiment():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({"error": "Invalid input, 'text' field is required."}), 400

    text = data['text']
    result = predict(model, text)
    
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
from flask import Flask, request, jsonify
from model import load_model, predict

app = Flask(__name__)
model = load_model()

@app.route("/predict", methods=["POST"])
def predict_route():
    data = request.get_json()
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "Champ 'text' manquant"}), 400
    result = predict(model, text)
    return jsonify({"text": text, **result})

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(debug=True)
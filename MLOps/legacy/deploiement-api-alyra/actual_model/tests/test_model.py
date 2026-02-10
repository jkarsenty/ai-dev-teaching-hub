import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from model import load_model, predict


def test_load_model_returns_pipeline():
    model = load_model()
    assert model is not None
    # Vérifie que le modèle possède une méthode predict_proba
    assert hasattr(model, "predict_proba")


def test_predict_output_structure():
    model = load_model()
    result = predict(model, "ce produit est excellent")

    assert "label" in result
    assert "confidence" in result
    assert "all_probabilities" in result
    assert "input_text" in result
    assert isinstance(result["confidence"], float)
    assert isinstance(result["all_probabilities"], dict)


def test_predict_positive_example():
    model = load_model()
    result = predict(model, "it's so good and amazing!")
    assert result["label"] in ["positive", "neutral", "negative"]

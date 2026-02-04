
import json, numpy as np, pickle

def model_fn(model_dir):
    with open(f"{model_dir}/heart_disease_model.pkl", 'rb') as f:
        return pickle.load(f)

def input_fn(request_body, content_type):
    if content_type == 'application/json':
        return json.loads(request_body)
    raise ValueError(f"Unsupported content type: {content_type}")

def predict_fn(input_data, model):
    features = np.array([
        input_data.get('Age', 0), input_data.get('Cholesterol', 0),
        input_data.get('BP', 0), input_data.get('Max HR', 0),
        input_data.get('ST depression', 0), input_data.get('Number of vessels fluro', 0),
        input_data.get('Chest pain type', 0), input_data.get('Exercise angina', 0)
    ])

    mean, std = np.array(model['feature_mean']), np.array(model['feature_std'])
    features_norm = (features - mean) / std

    w, b = np.array(model['weights']), model['bias']
    z = np.dot(w, features_norm) + b
    probability = 1 / (1 + np.exp(-z))
    prediction = int(probability >= 0.5)

    return {
        'prediction': prediction,
        'probability': float(probability),
        'risk_level': 'High Risk' if probability >= 0.5 else 'Low Risk'
    }

def output_fn(prediction, accept):
    if accept == 'application/json':
        return json.dumps(prediction), accept
    raise ValueError(f"Unsupported accept type: {accept}")

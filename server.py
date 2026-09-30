import os
import sys

# Ensure src directory is in sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from flask import Flask, request, jsonify
import torch

from generate import load_model_checkpoint, generate_text

app = Flask(__name__)

# Add CORS headers to allow requests from Vite frontend
@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response

# Lazy-load model instance
MODEL_CHECKPOINT = os.path.join(os.path.dirname(__file__), "models", "best_model.pt")
model, metadata, config, device = None, None, None, None

def get_model():
    global model, metadata, config, device
    if model is None:
        if os.path.exists(MODEL_CHECKPOINT):
            model, metadata, config, device = load_model_checkpoint(MODEL_CHECKPOINT)
        else:
            print(f"Warning: Checkpoint not found at {MODEL_CHECKPOINT}. Using default fallback.")
    return model, metadata, config, device

@app.route("/api/health", methods=["GET"])
def health_check():
    m, meta, cfg, dev = get_model()
    if m is not None:
        return jsonify({
            "status": "online",
            "company": "Rayvat Outsourcing - AI Labs",
            "model": "Word-Level PyTorch LSTM Text Generator",
            "vocab_size": meta.get("vocab_size", 0),
            "seq_length": meta.get("seq_length", 30),
            "device": str(dev)
        })
    return jsonify({"status": "model_not_ready", "company": "Rayvat Outsourcing"})

@app.route("/api/generate", methods=["POST", "OPTIONS"])
def api_generate():
    if request.method == "OPTIONS":
        return "", 200

    data = request.json or {}
    seed_text = data.get("seed", "to be or not")
    num_words = int(data.get("length", 80))
    temperature = float(data.get("temperature", 0.8))

    m, meta, cfg, dev = get_model()

    if m is None:
        return jsonify({
            "error": "Model checkpoint not found. Please train the model first.",
            "prompt": seed_text,
            "text": f"{seed_text} [Model weights training in background. Please wait a moment...]"
        }), 404

    generated_output = generate_text(
        model=m,
        metadata=meta,
        seed_text=seed_text,
        num_words=num_words,
        temperature=temperature,
        device=dev
    )

    return jsonify({
        "status": "success",
        "company": "Rayvat Outsourcing - AI Labs",
        "prompt": seed_text,
        "length": num_words,
        "temperature": temperature,
        "text": generated_output
    })

if __name__ == "__main__":
    print("==================================================")
    print(" RAYVAT OUTSOURCING - LSTM GENERATIVE AI BACKEND  ")
    print(" Running Flask API Server on http://localhost:5000 ")
    print("==================================================")
    app.run(host="0.0.0.0", port=5000, debug=False)

"""
Wildfire Detection Web Application
"""
import os
import sys
import json
import io
import base64
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import torch
from flask import Flask, render_template, request, jsonify

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

from src.data_loader import INPUT_FEATURES, FEATURE_INFO, NUM_CHANNELS, PATCH_SIZE
from src.models import get_model
from src.predict import predict_patch, generate_prediction_report

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024

def convert_numpy_types(obj):
    """Recursively convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, (np.integer, np.floating)):
        return float(obj) if isinstance(obj, np.floating) else int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

MODEL = None
DEVICE = None
MEANS = None
STDS = None
DATA_INFO = None

def load_model_once():
    global MODEL, DEVICE, MEANS, STDS, DATA_INFO
    if MODEL is not None:
        return

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    info_path = os.path.join(PROJECT_DIR, "checkpoints", "data_info.json")
    if os.path.exists(info_path):
        with open(info_path) as f:
            DATA_INFO = json.load(f)
        MEANS = np.array(DATA_INFO["means"], dtype=np.float32)
        STDS = np.array(DATA_INFO["stds"], dtype=np.float32)
    else:
        MEANS = np.zeros(NUM_CHANNELS, dtype=np.float32)
        STDS = np.ones(NUM_CHANNELS, dtype=np.float32)

    ckpt_path = os.path.join(PROJECT_DIR, "checkpoints", "best_model.pth")
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(PROJECT_DIR, "checkpoints", "final_model.pth")

    model, _ = get_model("unet_lite", in_channels=NUM_CHANNELS)
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"[webapp] Model loaded from {ckpt_path}")
    else:
        print("[webapp] WARNING: No checkpoint found, using random weights!")

    model = model.to(DEVICE)
    model.eval()
    MODEL = model

def fig_to_base64(fig):
    """Convert matplotlib figure to base64 PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return img_str

def create_feature_grid(patch):
    """Create a grid image showing all 12 feature channels."""
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    cmaps = ["RdYlGn", "terrain", "coolwarm", "Blues", "hot", "hot",
             "YlGnBu", "Blues", "RdBu_r", "YlOrRd", "Purples", "Reds"]

    for i, (feat, cmap) in enumerate(zip(INPUT_FEATURES, cmaps)):
        row, col = divmod(i, 4)
        ch = patch[i]
        im = axes[row, col].imshow(ch, cmap=cmap)
        info = FEATURE_INFO[feat]
        axes[row, col].set_title(f"{feat}\n({info['unit']})", fontsize=9)
        axes[row, col].axis("off")
        plt.colorbar(im, ax=axes[row, col], fraction=0.046, pad=0.04)

    fig.suptitle("12-Channel Satellite Feature Input", fontsize=14)
    plt.tight_layout()
    return fig_to_base64(fig)

def create_prediction_image(patch, prob_map, pred_mask):
    """Create the main prediction visualisation."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    cmap_fire = mcolors.ListedColormap(["#1a1a2e", "#e94560"])

    im = axes[0].imshow(patch[0], cmap="RdYlGn")
    axes[0].set_title("NDVI (Vegetation)", fontsize=11)
    axes[0].axis("off")
    plt.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)

    axes[1].imshow(patch[11], cmap=cmap_fire, vmin=0, vmax=1)
    axes[1].set_title("Previous Day Fire", fontsize=11)
    axes[1].axis("off")

    im = axes[2].imshow(prob_map, cmap="hot", vmin=0, vmax=1)
    axes[2].set_title("Fire Spread Probability", fontsize=11)
    axes[2].axis("off")
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    axes[3].imshow(pred_mask, cmap=cmap_fire, vmin=0, vmax=1)
    fire_px = int(pred_mask.sum())
    axes[3].set_title(f"Predicted Fire ({fire_px} pixels)", fontsize=11)
    axes[3].axis("off")

    plt.tight_layout()
    return fig_to_base64(fig)

@app.route("/")
def index():
    load_model_once()
    return render_template("index.html", feature_info=FEATURE_INFO,
                           input_features=INPUT_FEATURES)

@app.route("/about")
def about():
    return render_template("about.html", feature_info=FEATURE_INFO,
                           input_features=INPUT_FEATURES)

@app.route("/predict", methods=["POST"])
def predict():
    load_model_once()
    try:
        file = request.files.get("file")
        if file is None:
            return jsonify({"error": "No file uploaded"}), 400

        filename = file.filename.lower()

        if filename.endswith(".npy"):
            patch = np.load(io.BytesIO(file.read()))
        elif filename.endswith(".npz"):
            data = np.load(io.BytesIO(file.read()))
            for key in ["patch", "data", "X", "x", "input"]:
                if key in data:
                    patch = data[key]
                    break
            else:
                patch = data[list(data.keys())[0]]
        else:
            return jsonify({"error": "Unsupported format. Upload .npy or .npz file (shape: 12x64x64)"}), 400

        if patch.ndim == 3 and patch.shape == (PATCH_SIZE, PATCH_SIZE, NUM_CHANNELS):
            patch = patch.transpose(2, 0, 1)
        if patch.ndim != 3 or patch.shape != (NUM_CHANNELS, PATCH_SIZE, PATCH_SIZE):
            return jsonify({
                "error": f"Invalid shape {patch.shape}. Expected ({NUM_CHANNELS}, {PATCH_SIZE}, {PATCH_SIZE})"
            }), 400

        patch = patch.astype(np.float32)

        prob_map, pred_mask = predict_patch(MODEL, patch, MEANS, STDS, DEVICE)
        report = generate_prediction_report(patch, prob_map, pred_mask)

        prediction_img = create_prediction_image(patch, prob_map, pred_mask)
        feature_img = create_feature_grid(patch)

        return jsonify({
            "success": True,
            "prediction_image": prediction_img,
            "feature_image": feature_img,
            "report": convert_numpy_types(report),
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/demo", methods=["POST"])
def demo():
    """Run prediction on a random test sample (no upload needed)."""
    load_model_once()
    try:
        from src.data_loader import load_tfrecord_data
        X_test, y_test = load_tfrecord_data("test")

        fire_counts = (y_test == 1).sum(axis=(1, 2))
        has_fire = np.where(fire_counts > 10)[0]
        if len(has_fire) > 0:
            idx = np.random.choice(has_fire)
        else:
            idx = np.random.randint(len(X_test))

        patch = X_test[idx]
        gt = (y_test[idx] == 1).astype(np.float32)

        prob_map, pred_mask = predict_patch(MODEL, patch, MEANS, STDS, DEVICE)
        report = generate_prediction_report(patch, prob_map, pred_mask)
        report["ground_truth_fire_pixels"] = int(gt.sum())

        prediction_img = create_prediction_image(patch, prob_map, pred_mask)
        feature_img = create_feature_grid(patch)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        cmap_fire = mcolors.ListedColormap(["#1a1a2e", "#e94560"])
        axes[0].imshow(gt, cmap=cmap_fire, vmin=0, vmax=1)
        axes[0].set_title("Ground Truth")
        axes[0].axis("off")
        im = axes[1].imshow(prob_map, cmap="hot", vmin=0, vmax=1)
        axes[1].set_title("Predicted Probability")
        axes[1].axis("off")
        plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
        axes[2].imshow(pred_mask, cmap=cmap_fire, vmin=0, vmax=1)
        axes[2].set_title("Predicted Fire Mask")
        axes[2].axis("off")
        plt.tight_layout()
        comparison_img = fig_to_base64(fig)

        return jsonify({
            "success": True,
            "prediction_image": prediction_img,
            "feature_image": feature_img,
            "comparison_image": comparison_img,
            "report": convert_numpy_types(report),
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)

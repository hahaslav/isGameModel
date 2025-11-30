import marimo

__generated_with = "0.18.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Результати transformer моделі

    Валідаційна вибірка:
    - ROC AUC: 0.9949
    - Gini: 0.9899
    """)
    return


@app.cell(hide_code=True)
def _(live_refresh, mo, predict_image, take_screenshot):
    mo.md(rf"""
    {live_refresh}

    {predict_image(take_screenshot()):.2%}
    """)
    return


@app.cell
def _(load_image, os, predict_image):
    for image_name in os.listdir("new images"):
        predict_image(load_image(f"new images/{image_name}"))
    return


@app.cell
def _(ViTForImageClassification, ViTImageProcessor, sys):
    # 1. Configuration
    # Point this to the folder where you downloaded config.json and model.safetensors
    MODEL_PATH = "transformer" 

    # 2. Load Model and Processor
    # The library automatically detects you are on CPU
    try:
        processor = ViTImageProcessor.from_pretrained(MODEL_PATH)
        model = ViTForImageClassification.from_pretrained(MODEL_PATH)
    except OSError:
        print(f"Error: Could not find model files in {MODEL_PATH}")
        print("Make sure you downloaded the whole folder, including config.json")
        sys.exit(1)
    return model, processor


@app.cell
def _(Image):
    def load_image(image_path):
        try:
            # Open image
            image = Image.open(image_path)
            # Ensure it's RGB (removes Alpha channel if png has transparency)
            image = image.convert("RGB")
        except Exception as e:
            print(f"Error opening image: {e}")
            return
    
        print(f"Image: {image_path}")
        return image
    return (load_image,)


@app.cell
def _(model, processor, torch):
    def predict_image(image):
        # Preprocess (Resize, Normalize)
        inputs = processor(images=image, return_tensors="pt")

        # Inference
        # We use torch.no_grad() because we don't need to calculate gradients for training
        with torch.no_grad():
            outputs = model(**inputs)
    
        # Get probabilities
        logits = outputs.logits
        predicted_class_idx = logits.argmax(-1).item()
    
        # Get the label name (e.g., "work" or "game")
        predicted_label = model.config.id2label[predicted_class_idx]
    
        print(f"Prediction: {predicted_label.upper()}")
    
        # Optional: Print confidence score
        probs = torch.nn.functional.softmax(logits, dim=-1)
        confidence = probs[0][predicted_class_idx].item()
        print(f"Confidence: {confidence:.2%}")
        print("-" * 20)

        return probs[0][0]
    return (predict_image,)


@app.cell
def _(Image, mss):
    def take_screenshot():
        with mss() as sct:
            new_screenshot = sct.grab(sct.monitors[0])
        return Image.frombytes("RGB", new_screenshot.size, new_screenshot.bgra, "raw", "BGRX")
    return (take_screenshot,)


@app.cell
def _(mo):
    live_refresh = mo.ui.refresh(["1s", "3s"], label="Evaluate desktop in real time")
    return (live_refresh,)


@app.cell
def _():
    import torch
    from transformers import ViTForImageClassification, ViTImageProcessor
    from PIL import Image
    from mss.windows import MSS as mss
    import sys
    import os
    import marimo as mo
    return (
        Image,
        ViTForImageClassification,
        ViTImageProcessor,
        mo,
        mss,
        os,
        sys,
        torch,
    )


if __name__ == "__main__":
    app.run()

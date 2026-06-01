# Pneumonia Detector (chest X-ray CNN)

A convolutional neural network that looks at a chest X-ray and says whether it
shows pneumonia or a normal lung. Trained from scratch on grayscale X-rays with
3 conv blocks + a dense classifier.

> ⚠️ learning project, **not** a medical diagnosis tool.

## data

The classic Kaggle "Chest X-Ray Images (Pneumonia)" dataset works. Layout:

```
data/train/NORMAL/*.jpeg
data/train/PNEUMONIA/*.jpeg
data/val/NORMAL/*.jpeg
data/val/PNEUMONIA/*.jpeg
```

## run

```bash
pip install tensorflow

python train.py            # trains + saves pneumonia_model.keras
python predict.py xray.jpeg
```

tags: ai, ml, deep-learning, healthcare, tensorflow, cnn, medical-imaging

medical imaging + CNNs is one of the most genuinely useful uses of AI imo.

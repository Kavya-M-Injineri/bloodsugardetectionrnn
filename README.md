# Blood Sugar RNN Classifier 

> RNN-based model that reads blood sugar readings from uploaded images, with digit extraction and image verification — served via a Flask web application.

---

## Overview

This project combines computer vision and sequence modeling to automatically extract and classify blood sugar values from medical device images. An image verification step ensures only relevant uploads are processed, digits are extracted via OpenCV preprocessing, and an RNN model predicts the numeric reading — all accessible through a simple web interface.

---

## Features

- **Image Verification** — Filters out irrelevant uploads before processing.
- **Digit Extraction** — OpenCV-based preprocessing pipeline isolates numeric segments from the image.
- **RNN Inference** — Sequence model predicts digits from the extracted image features.
- **Flask Web App** — Upload an image and get an instant blood sugar reading in the browser.
- **Lazy Model Loading** — Model weights are loaded once on first request, keeping startup fast.

---

## Project Structure

```
blood-sugar-rnn/
├── app.py               # Flask app — handles uploads and prediction endpoint
├── rnn_model.py         # RNN model architecture and SEQ_LEN config
├── preprocess.py        # Digit extraction via OpenCV
├── verify_image.py      # Image relevance verification logic
├── train_model.py       # Model training script
├── model_weights.weights.h5  # Saved weights (git-ignored)
├── uploads/             # Temporary image storage (git-ignored)
├── templates/           # HTML templates for the web interface
└── requirements.txt     # Python dependencies
```

---

## Getting Started

### Prerequisites

- Python 3.8+

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Model

```bash
python train_model.py
```

Trains the RNN and saves weights to `model_weights.weights.h5`.

### 3. Launch the App

```bash
python app.py
```

Open your browser at **http://127.0.0.1:5000**

---

## API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/` | `GET` | Renders the upload interface |
| `/predict` | `POST` | Accepts an image file, returns predicted blood sugar value |

**Request** — `multipart/form-data` with a `file` field.

**Response (success)**
```json
{ "blood_sugar": "124" }
```

**Response (error)**
```json
{ "error": "Invalid image type. Please upload a blood sugar related image." }
```

---

## Pipeline

```
Uploaded Image → verify_image.py → preprocess.py → RNN Model → Blood Sugar Value
```

---

*No GPU required. Optimized for CPU inference.*

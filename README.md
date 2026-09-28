# ViT Image Classification — Streamlit

A college mini-project demonstrating image classification using the pre-trained `google/vit-base-patch16-224` Vision Transformer.

## Features
- Image upload
- Webcam/camera input
- Pre-trained ViT inference
- Confidence threshold slider
- Top-5 predictions
- Confidence visualization
- Streamlit Community Cloud deployment

## Run locally
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## GitHub structure
```text
vit-image-classification/
├── app.py
├── requirements.txt
└── README.md
```

Do not upload the downloaded Hugging Face model files; the app downloads them automatically.

## Streamlit Community Cloud
1. Push these files to a GitHub repository.
2. Sign in at https://share.streamlit.io/.
3. Create an app and select the repository and `main` branch.
4. Set the entrypoint to `app.py`.
5. Click Deploy.

The `requirements.txt` file tells Community Cloud which Python packages to install.

## Model
- Model: `google/vit-base-patch16-224`
- Task: Image classification
- Source: Hugging Face Model Hub
- License: Apache-2.0 (verify the model card for the exact version used)
- Model page: https://huggingface.co/google/vit-base-patch16-224

## Viva points
**Pre-trained model:** A model whose weights were learned during previous training.

**Inference:** Using learned weights to predict on new input.

**Confidence slider:** A user-defined threshold used to indicate whether the top prediction meets the selected confidence level; it does not change the model.

**Important:** This is image classification, not object detection, so it predicts a class for the image rather than drawing bounding boxes.

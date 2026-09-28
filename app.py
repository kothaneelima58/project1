import streamlit as st
import torch
import pandas as pd
from PIL import Image
from transformers import ViTImageProcessor, ViTForImageClassification

st.set_page_config(page_title='ViT Image Classifier', page_icon='🧠', layout='wide')
MODEL_NAME='google/vit-base-patch16-224'

@st.cache_resource
def load_model():
    processor=ViTImageProcessor.from_pretrained(MODEL_NAME)
    model=ViTForImageClassification.from_pretrained(MODEL_NAME)
    model.eval()
    return processor, model

processor, model=load_model()

def classify_image(image, top_k=5):
    image=image.convert('RGB')
    inputs=processor(images=image, return_tensors='pt')
    with torch.no_grad():
        outputs=model(**inputs)
    probs=torch.softmax(outputs.logits, dim=-1)
    values, indices=torch.topk(probs, top_k)
    return [{'label':model.config.id2label[i.item()], 'confidence':float(v.item())} for v,i in zip(values[0],indices[0])]

st.title('🧠 Image Classification using Vision Transformer')
st.caption('Pre-trained ViT • Image Upload • Webcam • Top-5 Predictions')
st.info('Uses the pre-trained google/vit-base-patch16-224 model. No model training is performed.')

with st.sidebar:
    st.header('⚙️ Settings')
    mode=st.radio('Choose input', ['Upload Image','Camera'])
    threshold=st.slider('Confidence threshold',0.0,1.0,0.50,0.05)
    top_k=st.slider('Number of predictions',1,5,5)
    st.divider()
    st.write('**Model:** google/vit-base-patch16-224')
    st.write('**Task:** Image Classification')

image=None
if mode=='Upload Image':
    f=st.file_uploader('📤 Upload an image',type=['jpg','jpeg','png','webp'])
    if f: image=Image.open(f).convert('RGB')
else:
    f=st.camera_input('📷 Take a picture')
    if f: image=Image.open(f).convert('RGB')

if image is not None:
    left,right=st.columns([1,1.25])
    with left:
        st.subheader('Input Image')
        st.image(image,use_container_width=True)
    with st.spinner('Running ViT inference...'):
        predictions=classify_image(image,top_k)
    top=predictions[0]
    with right:
        st.subheader('Prediction')
        st.metric('Predicted Class',top['label'])
        st.metric('Confidence',f"{top['confidence']*100:.2f}%")
        if top['confidence']>=threshold:
            st.success(f"Prediction meets the selected {threshold*100:.0f}% threshold.")
        else:
            st.warning(f"Prediction is below the selected {threshold*100:.0f}% threshold.")
        df=pd.DataFrame({'Class':[p['label'] for p in predictions], 'Confidence (%)':[p['confidence']*100 for p in predictions]})
        st.dataframe(df.style.format({'Confidence (%)':'{:.2f}'}),use_container_width=True,hide_index=True)
    st.subheader('📊 Prediction Confidence')
    chart=df.set_index('Class')
    st.bar_chart(chart,y='Confidence (%)')
    st.caption('Softmax probability is not a guarantee that the prediction is correct.')
else:
    st.subheader('👆 Upload an image or use the camera to start')
    st.write('The app preprocesses the image, runs the pretrained Vision Transformer, and displays the top predictions.')

st.divider()
with st.expander('📚 About this project'):
    st.write('**Workflow:** Input → Pre-processing → Pre-trained ViT → Inference → Softmax → Top Predictions → Visualization')
    st.write('**Model:** google/vit-base-patch16-224')
    st.markdown('[Hugging Face model page](https://huggingface.co/google/vit-base-patch16-224)')

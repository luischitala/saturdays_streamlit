import streamlit as st
from transformers import pipeline, ViTFeatureExtractor, ViTForImageClassification
from PIL import Image

# Análisis de sentimiento
st.header("Análisis de sentimientos")
pipe = pipeline('sentiment-analysis')
text = st.text_area("Ingrese un texto para analizar los sentimientos")

submitted = st.button("Submit")

if text:     
    if submitted:
        out = pipe(text)

        label = out[0]["label"]
        st.write(f'El resultado fue: {label}')
        st.json(out)

# Generador de textos
st.header("Generación de texto")
generator = pipeline("text-generation", model="distilgpt2")
text_gen = st.text_area("Ingrese un texto para completarlo")
submitted_gen = st.button("Submit gen")

if text_gen:     
    if submitted_gen:
        text_generated = generator(
                                   text_gen, 
                                   max_length=30,
                                   num_return_sequences=10)
        st.write(text_generated)

# Clasificación de imágenes
st.header("Clasificador de Imágenes")

feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

file = st.file_uploader("Subir Imágen", type=["jpg", "JPEG"])
submit_file = st.button("Enviar")
if submit_file:
    image = Image.open(file)
    inputs = feature_extractor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    # model predicts one of the 1000 ImageNet classes
    predicted_class_idx = logits.argmax(-1).item()
    st.write("Clase predicha:", model.config.id2label[predicted_class_idx])
    st.image(image)

# Traducción de texto

st.sidebar.title("Resumir Texto")

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
article = st.sidebar.text_area("Ingrese un texto para resumirlo")
resumir = st.sidebar.button("Translate")

if resumir and article: 
    article_sum = summarizer(article, max_length=130, min_length=30, do_sample=False)
    st.sidebar.write("Resultado:")
    st.sidebar.write(article_sum)
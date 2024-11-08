import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
from gtts import gTTS
import os


# Constants
max_length = 34
vocab_size = 8485
embedding_dim = 256

# Load Tokenizer
with open("tokenizer.pkl", "rb") as handle:
    tokenizer = pickle.load(handle)

# Load VGG16 Model for Feature Extraction
@st.cache_resource
def load_cnn_model():
    base_model = VGG16(weights='imagenet')
    cnn_model = Model(inputs=base_model.inputs, outputs=base_model.layers[-2].output)
    return cnn_model

# Load Caption Model
@st.cache_resource
def load_caption_model():
    inputs1 = Input(shape=(4096,))
    fe1 = Dropout(0.4)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)

    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, embedding_dim, mask_zero=True)(inputs2)
    se2 = Dropout(0.4)(se1)
    se3 = LSTM(256)(se2)

    decoder1 = add([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)

    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.load_weights("best_model.h5")
    return model

# Preprocess image for VGG16
def preprocess_image(image):
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    image = image.resize((224, 224))
    image_array = np.array(image)
    image_array = preprocess_input(image_array)
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

# Generate caption for the image
def generate_caption(image_features, caption_model):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = caption_model.predict([image_features, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = tokenizer.index_word.get(yhat)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'endseq':
            break
    caption = in_text.replace('startseq', '').replace('endseq', '').strip()
    return caption

# Convert caption text to speech
def text_to_speech(text):
    tts = gTTS(text=text, lang='en')
    audio_file = "caption_audio.mp3"
    tts.save(audio_file)
    return audio_file

# CSS styles
st.markdown("""
    <style>
    /* Header */
    .title-container {
        text-align: center;
        font-size: 36px;
        color: #4A90E2;
        font-weight: bold;
        padding: 20px;
    }
    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #6E757C;
    }

    /* File uploader */
    .stFileUploader {
        border: 2px solid #4A90E2;
        border-radius: 10px;
        padding: 20px;
    }

    /* Caption output */
    .caption-output {
        font-size: 24px;
        font-weight: bold;
        color: #333;
        text-align: center;
        margin-top: 20px;
    }

    /* Button */
    .stButton button {
        background-color: #4A90E2;
        color: white;
        font-size: 16px;
        padding: 10px 20px;
        border: none;
        border-radius: 5px;
    }
    </style>
""", unsafe_allow_html=True)

# Streamlit layout without the custom icon
st.markdown("<div class='title-container'>Image Caption Generator</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Upload an image, generate a caption, and listen to the AI-generated caption!</div>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    if st.button("Generate Caption 🎉", key="caption_button"):
        st.write("Generating caption...")
        
        cnn_model = load_cnn_model()
        caption_model = load_caption_model()

        preprocessed_image = preprocess_image(image)
        image_features = cnn_model.predict(preprocessed_image).reshape((1, 4096))

        caption = generate_caption(image_features, caption_model)
        st.markdown(f"<div class='caption-output'>Caption: {caption}</div>", unsafe_allow_html=True)

        # Generate and play audio for the caption
        audio_file = text_to_speech(caption)
        audio_bytes = open(audio_file, "rb").read()
        st.audio(audio_bytes, format="audio/mp3")

        # Clean up audio file
        os.remove(audio_file)

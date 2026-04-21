import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions

# Load model
model = tf.keras.applications.MobileNetV2(weights="imagenet")

# Title
st.title("♻️ AI Waste Classifier")

# Preprocess image
def preprocess(img):
    img = img.resize((224, 224))
    img = np.array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img


# YOUR FUNCTION (IMPROVED VERSION)
def classify_waste(decoded):

    organic_keywords = [
        "banana","apple","orange","mango","grape","strawberry","pineapple",
        "watermelon","papaya","pear","peach","plum","kiwi","lemon","lime",
        "carrot","potato","tomato","onion","garlic","cucumber","cabbage",
        "broccoli","spinach","lettuce","beans","peas","corn",
        "pizza","burger","sandwich","rice","pasta","bread","cake","food"
    ]

    plastic_keywords = [
        "bottle","plastic","cup","container","jar","bag","can","lid",
        "water bottle","soda bottle"
    ]

    paper_keywords = [
        "paper","carton","box","cardboard","tissue","book","newspaper"
    ]

    organic_score = 0
    plastic_score = 0
    paper_score = 0

    for item in decoded:
        label = item[1].lower()
        conf = item[2]

        for k in organic_keywords:
            if k in label:
                organic_score += conf

        for k in plastic_keywords:
            if k in label:
                plastic_score += conf

        for k in paper_keywords:
            if k in label:
                paper_score += conf

    # FINAL DECISION
    if organic_score >= plastic_score and organic_score >= paper_score and organic_score > 0:
        return "🌱 Organic Waste"

    elif plastic_score >= organic_score and plastic_score >= paper_score and plastic_score > 0:
        return "♻️ Plastic Waste"

    elif paper_score > 0:
        return "📄 Paper Waste"

    else:
        return "🗑️ General Waste"


# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)

    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    processed_img = preprocess(img)

    # Predict
    preds = model.predict(processed_img)
    decoded = decode_predictions(preds, top=5)[0]

    # DEBUG (VERY IMPORTANT)
    st.write("Detected labels:", [item[1] for item in decoded])

    # Classify
    result = classify_waste(decoded)

    # Output
    st.subheader("Result")
    st.success(result)
import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
import pandas as pd
from PIL import Image
from difflib import get_close_matches

# Load Pre-trained Model
@st.cache_resource
def load_model():
    return MobileNetV2(weights='imagenet')

model = load_model()

# Load Nutritional Database
@st.cache_data
def load_nutritional_database():
    data = {
        'Food': ['apple', 'banana', 'pizza', 'hamburger', 'salad'],
        'Calories': [95, 105, 285, 354, 152],
        'Sodium_mg': [2, 1, 640, 970, 150]
    }
    return pd.DataFrame(data)

# Match predicted food to database
def match_food_label(pred_label, db):
    food_list = db['Food'].str.lower().tolist()
    match = get_close_matches(pred_label.lower(), food_list, n=1, cutoff=0.4)
    return match[0] if match else None

# Recognize Food from Image
def recognize_food(uploaded_img):
    img = uploaded_img.resize((224, 224))
    x = np.array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x)
    decoded_preds = decode_predictions(preds, top=3)[0]
    return [(label, name.replace("_", " "), conf) for (label, name, conf) in decoded_preds]

# Rule-Based Dietary Recommendation
def rule_based_recommendation(matched_food, db):
    sodium_limit = 500
    food_info = db[db['Food'].str.lower() == matched_food]
    if food_info.empty:
        return f"No nutritional data for {matched_food}. Please review manually."
    
    food_info = food_info.iloc[0]
    if food_info['Sodium_mg'] > sodium_limit:
        return f"Warning: {matched_food.capitalize()} is high in sodium!"
    else:
        return f"{matched_food.capitalize()} is suitable for your diet."

# Streamlit App
def main():
    st.title("AI Dietary Recommendation System")
    st.write("Upload an image of your food and get personalized dietary advice.")

    uploaded_file = st.file_uploader("Choose a food image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption='Uploaded Image', use_column_width=True)

        with st.spinner("Analyzing image..."):
            predictions = recognize_food(img)

        st.subheader("Top Predictions:")
        for label, name, conf in predictions:
            st.write(f"{name.capitalize()}** - {conf * 100:.2f}% confidence")

        top_pred = predictions[0][1]
        db = load_nutritional_database()
        matched_food = match_food_label(top_pred, db)

        if matched_food:
            recommendation = rule_based_recommendation(matched_food, db)
            st.info(recommendation)
        else:
            st.warning(f"No match found for '{top_pred}' in the database. Please review manually.")

if __name__ == "__main__":
    main()


import streamlit as st
import pickle
import re
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

model = pickle.load(open("model/spam_model.pkl", "rb"))
vectorizer = pickle.load(open("model/vectorizer.pkl", "rb"))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

def predict(message):
    vector = vectorizer.transform([clean_text(message)])
    result = model.predict(vector)[0]
    return "Spam" if result == 1 else "Not Spam"

st.title("📧 Email Spam Detector")

message = st.text_area("Enter your message")

if st.button("Check"):
    if message:
        st.write(predict(message))
import pickle
import re

# -------------------------------
# Load saved model and vectorizer
# -------------------------------
model = pickle.load(open("model/spam_model.pkl", "rb"))
vectorizer = pickle.load(open("model/vectorizer.pkl", "rb"))

# -------------------------------
# Text cleaning function
# -------------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = text.strip()
    return text

# -------------------------------
# Prediction function
# -------------------------------
def predict_spam(message):
    message = clean_text(message)
    vectorized = vectorizer.transform([message])
    prediction = model.predict(vectorized)
    
    return "Spam" if prediction[0] == 1 else "Not Spam"

# -------------------------------
# User input
# -------------------------------
if __name__ == "__main__":
    print("\n Email Spam Detector")
    print("---------------------------")
    
    user_input = input("Enter your email message: ")
    
    result = predict_spam(user_input)
    
    print("\nPrediction:", result)
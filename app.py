from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import string
import nltk

# Ensure NLTK stopwords are downloaded
nltk.download('stopwords')
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

# Load the saved TF-IDF vectorizer and Logistic Regression model.
vectorizer = joblib.load('tfidf_vectorizer.pkl')
model = joblib.load('logistic_model.pkl')

def predict_category(text):
    clean_text = preprocess_text(text)
    text_vect = vectorizer.transform([clean_text])
    prediction = model.predict(text_vect)
    return prediction[0]

# Create FastAPI app
app = FastAPI(title="Text Classification API")

# Define a request model using Pydantic
class TextData(BaseModel):
    inquiry: str

# Create an endpoint to predict the category
@app.post("/predict")
def get_prediction(data: TextData):
    category = predict_category(data.inquiry)
    return {"predicted_category": category}



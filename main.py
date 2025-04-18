import joblib
import string
import nltk
from nltk.corpus import stopwords

# Ensure the NLTK stopwords resource is available.
nltk.download('stopwords')

# Load stopwords.
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    """
    Preprocess the input text by converting to lowercase, removing punctuation, and eliminating stopwords.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()  # Lowercase the text.
    text = "".join(ch for ch in text if ch not in string.punctuation)  # Remove punctuation.
    words = text.split()
    words = [word for word in words if word not in stop_words]  # Remove stopwords.
    return " ".join(words)

def load_model_and_vectorizer():
    """
    Load the pre-trained TF-IDF vectorizer and Logistic Regression model using joblib.
    """
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    model = joblib.load('logistic_model.pkl')
    return vectorizer, model

def predict_category(text, vectorizer, model):
    """
    Predict the category for the input text.
    """
    clean_text = preprocess_text(text)
    text_vect = vectorizer.transform([clean_text])
    prediction = model.predict(text_vect)
    return prediction[0]

if __name__ == '__main__':
    # Load the model and vectorizer.
    vectorizer, model = load_model_and_vectorizer()
    
    # Prompt the user to enter a text inquiry.
    user_input = input("Enter your inquiry: ")
    
    # Predict and display the category.
    result = predict_category(user_input, vectorizer, model)
    print(f"Predicted Category: {result}")

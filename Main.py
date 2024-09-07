import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import streamlit as st
import nltk

#stopwords
nltk.download('stopwords')
nltk.download('punkt')

stop_words = set(stopwords.words('english'))

@st.cache_data
def load_data():
    data = pd.read_csv('archive/IMDB Dataset.csv')
    return data

@st.cache_data
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    words = word_tokenize(text)
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

@st.cache_data
def train_model(data):
    data['clean_review'] = data['review'].apply(preprocess_text)
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(data['clean_review'])
    y = data['sentiment']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = MultinomialNB()
    model.fit(X_train, y_train)
    return model, vectorizer

# Load data
data = load_data()
model, vectorizer = train_model(data)

# UI
st.title('Sentiment Analysis Tool')

user_input = st.text_area('Enter text for sentiment analysis')

if st.button('Analyze'):
    processed_text = preprocess_text(user_input)
    vectorized_text = vectorizer.transform([processed_text])
    prediction = model.predict(vectorized_text)
    st.write('Predicted Sentiment:', prediction[0])

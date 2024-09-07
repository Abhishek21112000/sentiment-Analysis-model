
import pandas as pd
import spacy
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import streamlit as st
import os

#streramlit
st.set_page_config(page_title='Sentiment Analysis Tool', layout='wide')

# Load SpaCy
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner', 'textcat'])

# Cache
stop_words = nlp.Defaults.stop_words

@st.cache_data
def load_data():
    #need to make a csv file more stable so that the number of row and colums are same for each file 
    """
    file_1 = pd.read_csv('archive(IMDB)\IMDB Dataset.csv')
    file_2 = pd.read_csv('archive(twitter)\witter_training.csv')
    data = pd.concat([file_1, file_2])
    """
    data = pd.read_csv('archive(IMDB)\IMDB Dataset.csv')
    return data

def preprocess_text(text):
    doc = nlp(text.lower())  # Tokenize text
    tokens = [token.text for token in doc if not token.is_stop and token.is_alpha]
    return ' '.join(tokens)

def train_and_save_model(data):
    data['clean_review'] = data['review'].apply(preprocess_text)
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(data['clean_review'])
    y = data['sentiment']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = MultinomialNB()
    model.fit(X_train, y_train)
    
    # Save model and vectorizer
    with open('sentiment_model.pkl', 'wb') as model_file:
        pickle.dump(model, model_file)
    with open('vectorizer.pkl', 'wb') as vec_file:
        pickle.dump(vectorizer, vec_file)

def load_model_and_vectorizer():
    with open('sentiment_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('vectorizer.pkl', 'rb') as vec_file:
        vectorizer = pickle.load(vec_file)
    return model, vectorizer

# Check model
if not os.path.exists('sentiment_model.pkl') or not os.path.exists('vectorizer.pkl'):
    # train if not there
    data = load_data()
    train_and_save_model(data)

# Load model and vectorizer
model, vectorizer = load_model_and_vectorizer()

# UI
st.title('Sentiment Analysis Tool')
st.markdown("### Analyze movie reviews and get sentiment predictions")

# Layout
col1, col2 = st.columns([2, 1])

with col1:
    user_input = st.text_area('Enter text for sentiment analysis', height=250)

with col2:
    if st.button('Analyze'):
        if user_input:
            processed_text = preprocess_text(user_input)
            vectorized_text = vectorizer.transform([processed_text])
            prediction = model.predict(vectorized_text)
            st.write(f'**Predicted Sentiment:** {prediction[0]}')
        else:
            st.warning("Please enter some text for analysis.")





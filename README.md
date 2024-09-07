# sentiment-Analysis-model

# Sentiment Analysis Tool

This repository contains a simple web application that performs sentiment analysis on movie reviews. The tool is built using Streamlit, SpaCy, Scikit-learn, and Naive Bayes classification. The application loads a pre-trained sentiment analysis model and allows users to input text for sentiment prediction.

## Features
- **Sentiment Analysis**: Analyze movie reviews or any text and get predictions of the sentiment (positive or negative).
- **Preprocessing**: The text is preprocessed by tokenizing, removing stopwords, and converting it to lowercase.
- **Model**: Uses a Naive Bayes classifier trained on IMDB movie review data.
- **Web Interface**: A simple UI built with Streamlit for easy interaction.

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/sentiment-analysis-tool.git
   cd sentiment-analysis-tool
   ```

2. **Install the required libraries**:
   Create and activate a virtual environment (optional but recommended).
   
   python -m venv venv
   source venv/bin/activate  
   
   
   Install the dependencies:
   
   pip install -r requirements.txt
   

3. **Download SpaCy language model**:
   
   python -m spacy download en_core_web_sm
   

4. **Run the Streamlit application**:
   
   streamlit run app.py


## How It Works

1. **Loading Data**: The tool reads the IMDB dataset from a CSV file and preprocesses the text.
2. **Preprocessing**: The input text is tokenized using SpaCy, stopwords are removed, and the text is converted to lowercase.
3. **Training the Model** (if no model exists):
   - A Naive Bayes classifier is trained using TF-IDF vectorized movie reviews.
   - The model and vectorizer are saved as pickle files (`sentiment_model.pkl` and `vectorizer.pkl`).
4. **Sentiment Prediction**:
   - When the user inputs text, it is preprocessed and transformed using the saved TF-IDF vectorizer.
   - The pre-trained Naive Bayes classifier predicts the sentiment (positive or negative) of the input text.

## Files

- **app.py**: Main application file with Streamlit UI and model logic.
- **requirements.txt**: Lists all dependencies required to run the project.
- **sentiment_model.pkl**: Pickle file storing the trained Naive Bayes sentiment analysis model.
- **vectorizer.pkl**: Pickle file storing the TF-IDF vectorizer.

## Usage

1. After running the Streamlit application, open the provided local URL in a browser.
2. Enter the text you want to analyze in the input box.
3. Click the "Analyze" button to get the sentiment prediction.

## Dependencies

- pandas
- spacy
- scikit-learn
- streamlit
- pickle

You can install these dependencies via the `requirements.txt` file:


pip install -r requirements.txt


## Data
The tool uses the IMDB movie review dataset for training the model. In the current state, it only uses the IMDB dataset, but the functionality to combine datasets is available.

## Future Improvements

- Add more datasets (like Twitter reviews) for broader sentiment analysis.
- Improve the UI for better user experience.
- Optimize model performance with more advanced techniques.
- Add support for multi-class sentiment analysis.





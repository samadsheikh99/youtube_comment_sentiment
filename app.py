from flask import Flask, request, render_template
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


app = Flask(__name__)

# Download NLTK stopwords (if not already downloaded)
nltk.download('stopwords')

# Load the pre-trained models
with open('count_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

with open('random_forest_model.pkl', 'rb') as f:
    rf_model = pickle.load(f)

# Function to preprocess the text
def preprocess_text(text):
    # Remove punctuation using regex
    text = re.sub(r'[^\w\s]', '', text)  # Keep only alphanumeric and whitespace
    # Convert to lowercase
    text = text.lower()
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]
    # Stemming
    stemmer = PorterStemmer()
    stemmed_words = [stemmer.stem(word) for word in filtered_words]
    return ' '.join(stemmed_words)

# Function to make predictions
def make_predictions(text):
    cleaned_text = preprocess_text(text)
    text_counts = vectorizer.transform([cleaned_text])
    prediction = rf_model.predict(text_counts)
    predicted_label = label_encoder.inverse_transform(prediction)[0]
    return predicted_label

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        comment = request.form['comment']
        sentiment = make_predictions(comment)
        return render_template('index.html', sentiment=sentiment, comment=comment)
    return render_template('index.html', sentiment=None, comment=None)

if __name__ == '__main__':
    app.run(debug=True,host="0.0.0.0", port=5000)
from flask import Flask, request, render_template
import re
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from googletrans import Translator
import nltk

# Download stopwords
nltk.download('stopwords')

app = Flask(__name__)

# Load model and vectorizer
model = joblib.load('logistic_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# NLP setup
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))
translator = Translator()

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None

    if request.method == 'POST':
        user_input = request.form['news'].strip()

        try:
            # Detect language
            detected_lang = translator.detect(user_input).lang

            # Translate to English if needed
            if detected_lang != 'en':
                user_input = translator.translate(user_input, dest='en').text

            # Preprocess
            user_input = re.sub('[^a-zA-Z]', ' ', user_input)
            user_input = user_input.lower()
            words = user_input.split()
            processed_input = ' '.join(ps.stem(word) for word in words if word not in stop_words)

            # Vectorize
            input_vector = vectorizer.transform([processed_input])

            # Predict
            result = model.predict(input_vector)
            prediction = "Fake News" if result[0] == 1 else "Real News"

        except Exception as e:
            prediction = f"Error processing input: {e}. Please enter valid content."

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

# save_model.py

import pandas as pd
import re
import joblib
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import nltk

nltk.download('stopwords')

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = re.sub('[^a-zA-Z]', ' ', text).lower()
    words = text.split()
    return ' '.join(ps.stem(word) for word in words if word not in stop_words)

# Load your dataset (update path accordingly)
df = pd.read_csv('train.csv', on_bad_lines='skip')
df.fillna('', inplace=True)

if 'author' in df.columns and 'title' in df.columns:
    df['content'] = df['author'] + ' ' + df['title']
else:
    df['content'] = df['text']  # fallback

df['content'] = df['content'].apply(preprocess_text)

X = df['content']
y = df['label']

vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, stratify=y, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

# Save the model and vectorizer
joblib.dump(model, 'logistic_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')

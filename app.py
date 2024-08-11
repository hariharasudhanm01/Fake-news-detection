from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

# Load dataset
News = pd.read_csv('train.csv')

# Preprocessing
News = News.fillna(' ')
News['content'] = News['author'] + " " + News['title']

# Stemming process
ps = PorterStemmer()

def stemming(content):
    stemmed_content = content.lower()
    stemmed_content = re.sub(r'\W', ' ', stemmed_content)  # Remove special characters
    stemmed_content = re.sub(r'\s+', ' ', stemmed_content)  # Remove extra spaces
    stemmed_content = stemmed_content.split()
    stemmed_content = [ps.stem(word) for word in stemmed_content if word not in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content

News['content'] = News['content'].apply(stemming)

# Separating the data and label
X = News['content'].values
y = News['label'].values

# Converting the textual data to numerical data
vector = TfidfVectorizer()
vector.fit(X)
X = vector.transform(X)

# Training the Model using Logistic Regression
model = LogisticRegression()
model.fit(X, y)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/check_news', methods=['POST'])
def check_news():
    news = request.form['news']
    # Preprocess the input news
    news = stemming(news)
    # Vectorize the input news
    news_vectorized = vector.transform([news])
    # Predict
    prediction = model.predict(news_vectorized)
    # Return the result
    if prediction[0] == 1:
        result = 'Fake news'
    else:
        result = 'Real news'
    return jsonify({'result': result})

if __name__ == '__main__':
    app.run(debug=True)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
News = pd.read_csv('train.csv')
print(News.head())

# Preprocessing
print(News.shape)
print(News.isna().sum())
News = News.fillna(' ')
print(News.isna().sum())

# Combine author and title to create the content
News['content'] = News['author'] + " " + News['title']
print(News.head())
print(News['content'][20796])

# Importing NLTK
nltk.download('stopwords')

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
print(News['content'])

# Separating the data and label
X = News['content'].values
y = News['label'].values
print(X)

# Converting the textual data to numerical data
vector = TfidfVectorizer()
vector.fit(X)
X = vector.transform(X)
print(X)

# Visualizing the distribution of labels
labels_counts = News['label'].value_counts()
labels = ['Real News', 'Fake News']
plt.figure(figsize=(8, 8))
plt.pie(labels_counts, labels=labels, autopct='%1.1f%%', startangle=140)
plt.title("Distribution of Real and Fake News")
plt.show()

plt.figure(figsize=(8, 6))
sns.countplot(x=News['label'])
plt.xticks([0, 1], ['Real News', 'Fake News'])
plt.title("Count of Real and Fake News")
plt.xlabel("News Type")
plt.ylabel("Count")
plt.show()

# Splitting the dataset to training & test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=1)
print(X_train.shape)
print(X_test.shape)

# Training the Model using Logistic Regression
model = LogisticRegression()
model.fit(X_train, y_train)

train_y_pred = model.predict(X_train)
print("Train accuracy:", accuracy_score(train_y_pred, y_train))

test_y_pred = model.predict(X_test)
print("Test accuracy:", accuracy_score(test_y_pred, y_test))

# Feature importance
feature_names = vector.get_feature_names_out()
coefficients = model.coef_.flatten()
feature_importance = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients})
feature_importance = feature_importance.sort_values(by='Coefficient', ascending=False).head(20)

plt.figure(figsize=(10, 6))
sns.barplot(x='Coefficient', y='Feature', data=feature_importance)
plt.title("Top 20 Important Features for Fake News Detection")
plt.xlabel("Coefficient Value")
plt.ylabel("Feature")
plt.show()

# Testing with a sample input from the test set
input_data = X_test[45]
prediction = model.predict(input_data)
if prediction[0] == 1:
    print('Fake news')
else:
    print('Real news')
print(News['content'].iloc[45])
  
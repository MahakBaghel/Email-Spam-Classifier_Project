import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import string

# NLP
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# ML
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score

# DOWNLOAD NLTK DATA (Run once)
nltk.download('stopwords')

# LOAD DATASETS
df = pd.read_csv(r"D:\Email_Spam_Classifier_Project\Dataset\spam_or_not_spam.csv", encoding='latin-1')

# Rename columns properly
df.columns = ['email', 'label']

# HANDLE NULL VALUES
df.dropna(inplace=True)

# FIX LABEL ISSUE
# -----------------------------
# Your dataset already has 0 and 1 (not ham/spam)
# So DO NOT map again
# Just ensure integer type
df['label'] = df['label'].astype(int)
# TEXT CLEANING FUNCTION
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))  # optimize

def clean_text(text):
    text = str(text).lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    words = text.split()
    words = [
        ps.stem(word)
        for word in words
        if word not in stop_words
    ]
    
    return " ".join(words)

# Apply cleaning
df['cleaned'] = df['email'].apply(clean_text)

# FEATURE EXTRACTION (TF-IDF)
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['cleaned'])   
y = df['label']

# TRAIN TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# MODEL TRAINING
model = MultinomialNB()
model.fit(X_train, y_train)

# MODEL EVALUATION
y_pred = model.predict(X_test)

print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall   :", recall_score(y_test, y_pred))

# SAVE MODEL & VECTORIZER
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("✅ Model and vectorizer saved successfully!")
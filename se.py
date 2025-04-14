import re
import sys
import pickle
import random
import pandas as pd
import numpy as np
import nltk
from nltk import FreqDist
from collections import Counter
from scipy.sparse import lil_matrix
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

# Load dataset and drop rows with missing text
data = pd.read_csv(r"C:\01_College\SEM 6\Datasets\Tweets.csv").dropna(subset=['text'])

class SentimentAnalyzer:
    def __init__(self):
        # Configuration
        self.UNIGRAM_SIZE = 1000  # Reduced for small dataset
        self.VOCAB_SIZE = self.UNIGRAM_SIZE
        self.FEAT_TYPE = 'frequency'  # 'presence' or 'frequency'
        
        # Data structures
        self.unigrams = None
        self.label_encoder = LabelEncoder()
        
        # Models
        self.models = {
            'naive_bayes': None,
            'logistic': None,
            'svm': None
        }
        
    def preprocess_word(self, word):
        word = word.strip('\'"?!,.():;')
        word = re.sub(r'(.)\1+', r'\1\1', word)
        word = re.sub(r'(-|\')', '', word)
        return word

    def is_valid_word(self, word):
        return (re.search(r'^[a-zA-Z][a-z0-9A-Z\._]*$', word) is not None)

    def handle_emojis(self, tweet):
        tweet = re.sub(r'(:\s?\)|:-\)|\(\s?:|\(-:|:\'\))', ' EMO_POS ', tweet)
        tweet = re.sub(r'(:\s?D|:-D|x-?D|X-?D)', ' EMO_POS ', tweet)
        tweet = re.sub(r'(<3|:\*)', ' EMO_POS ', tweet)
        tweet = re.sub(r'(;-?\)|;-?D|\(-?;)', ' EMO_POS ', tweet)
        tweet = re.sub(r'(:\s?\(|:-\(|\)\s?:|\)-:)', ' EMO_NEG ', tweet)
        tweet = re.sub(r'(:,\(|:\'\(|:"\()', ' EMO_NEG ', tweet)
        return tweet

    def preprocess_tweet(self, tweet):
        # Handle NaN values
        if pd.isna(tweet):
            return ""
            
        processed_tweet = []
        tweet = str(tweet).lower()  # Convert to string in case it's not
        tweet = re.sub(r'((www\.[\S]+)|(https?://[\S]+))', ' URL ', tweet)
        tweet = re.sub(r'@[\S]+', 'USER_MENTION', tweet)
        tweet = re.sub(r'#(\S+)', r' \1 ', tweet)
        tweet = re.sub(r'\brt\b', '', tweet)
        tweet = re.sub(r'\.{2,}', ' ', tweet)
        tweet = tweet.strip(' "\'')
        tweet = self.handle_emojis(tweet)
        tweet = re.sub(r'\s+', ' ', tweet)
        words = tweet.split()

        for word in words:
            word = self.preprocess_word(word)
            if self.is_valid_word(word):
                processed_tweet.append(word)

        return ' '.join(processed_tweet)

    def process_data(self, df):
        processed_data = []
        all_words = []
        
        for _, row in df.iterrows():
            text_id = row['textID']
            text = row['text']
            sentiment = row['sentiment']
            
            processed_text = self.preprocess_tweet(text)
            words = processed_text.split()
            all_words.extend(words)
            
            processed_data.append({
                'text_id': text_id,
                'text': processed_text,
                'sentiment': sentiment,
                'words': words
            })
        
        # Create vocabulary
        freq_dist = FreqDist(all_words)
        self.unigrams = {word: i for i, (word, count) 
                        in enumerate(freq_dist.most_common(self.UNIGRAM_SIZE))}
        
        return processed_data

    def get_feature_vector(self, words):
        feature_vector = []
        for word in words:
            if word in self.unigrams:
                feature_vector.append(word)
        return feature_vector

    def extract_features(self, data):
        features = lil_matrix((len(data), self.VOCAB_SIZE))
        labels = []
        
        for i, item in enumerate(data):
            words = item['words']
            sentiment = item['sentiment']
            
            # Get feature vector
            feature_vector = self.get_feature_vector(words)
            
            # Create feature row
            if self.FEAT_TYPE == 'presence':
                feature_vector = set(feature_vector)
                
            for word in feature_vector:
                idx = self.unigrams[word]
                features[i, idx] += 1
            
            labels.append(sentiment)
        
        # Encode labels
        labels = self.label_encoder.fit_transform(labels)
        return features, labels

    def train_models(self, X_train, y_train):
        # Naive Bayes
        print("\nTraining Naive Bayes...")
        nb = MultinomialNB()
        nb.fit(X_train, y_train)
        self.models['naive_bayes'] = nb
        
        # Logistic Regression
        print("Training Logistic Regression...")
        lr = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
        lr.fit(X_train, y_train)
        self.models['logistic'] = lr
        
        # SVM
        print("Training SVM...")
        svm = LinearSVC(C=0.1, multi_class='ovr')
        svm.fit(X_train, y_train)
        self.models['svm'] = svm

    def evaluate_models(self, X_test, y_test):
        results = {}
        for name, model in self.models.items():
            pred = model.predict(X_test)
            acc = accuracy_score(y_test, pred)
            results[name] = acc
            print(f"{name} accuracy: {acc:.2f}")
        return results

    def show_sample_predictions(self, X_test, y_test, processed_data):
        print("\nSample predictions:")
        # Get the number of rows in the sparse matrix correctly
        n_samples = X_test.shape[0]
        sample_idx = random.sample(range(n_samples), min(5, n_samples))
        for idx in sample_idx:
            matching_items = [d for d in processed_data if self.label_encoder.transform([d['sentiment']])[0] == y_test[idx]]
            if matching_items:
                text = matching_items[0]['text']
                true_label = self.label_encoder.inverse_transform([y_test[idx]])[0]
                print(f"\nText: {text}")
                print(f"True sentiment: {true_label}")
                for name, model in self.models.items():
                    pred = model.predict(X_test[idx].reshape(1, -1))
                    pred_label = self.label_encoder.inverse_transform(pred)[0]
                    print(f"{name} prediction: {pred_label}")

    def run(self):
        # Process data
        print("Preprocessing data...")
        processed_data = self.process_data(data)
        
        # Extract features
        print("Extracting features...")
        X, y = self.extract_features(processed_data)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)
        
        # Train models
        print("\nTraining models...")
        self.train_models(X_train, y_train)
        
        # Evaluate
        print("\nEvaluating models...")
        results = self.evaluate_models(X_test, y_test)
        
        # Show sample predictions
        self.show_sample_predictions(X_test, y_test, processed_data)

if __name__ == '__main__':
    analyzer = SentimentAnalyzer()
    analyzer.run()
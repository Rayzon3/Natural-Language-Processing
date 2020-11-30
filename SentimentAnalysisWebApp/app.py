from flask import Flask, render_template, flash, request, url_for, redirect
import numpy as np
from numpy import array
import pandas as pd
import re
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import pickle

app = Flask(__name__)


global model, maxLength
model = load_model('Sentiment_LSTM_model.h5')    
maxLength = 100;

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

@app.route("/", methods=['GET', 'POST'])
def home():
    return "hello"


@app.route("/sentiment_prediction", methods=['GET', 'POST'])
def prediction():
    if request.method == 'POST':
        text = request.form['sentence']
        sentiment = ''
        # Tokenize text
        x_test = pad_sequences(tokenizer.texts_to_sequences([text]), maxlen=maxLength)
        # Predict
        score = model.predict([x_test])[0]
        if(score <= 0.5):
            sentiment = "Negative"
        if(score >= 0.5):
            sentiment = "Positive"
        return render_template("prediction.html", sentiment = sentiment)
    
    else:
        return render_template("form.html")
    

if __name__ == "__main__":
    app.run(debug=True)
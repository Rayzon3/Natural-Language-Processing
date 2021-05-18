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


global model_CNN, model_LSTM, maxLength
model_CNN = load_model('cnn_model.h5') 
model_LSTM = load_model('Sentiment_LSTM_model.h5') 
maxLength = 100;

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

@app.route("/", methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        redirect("form.html")
    return render_template("index.html")

@app.route("/formCNN", methods=['GET', 'POST'])
def form1():
    return render_template("formCNN.html")

@app.route("/formLSTM", methods=['GET', 'POST'])
def form2():
    return render_template("formLSTM.html")


@app.route("/sentiment_prediction_CNN", methods=['GET', 'POST'])
def prediction_CNN():
    if request.method == 'POST':
        text = request.form['sentence']
        sentiment = ''
        # Tokenize text
        x_test = pad_sequences(tokenizer.texts_to_sequences([text]), maxlen=maxLength)
        # Predict
        score = model_CNN.predict([x_test])[0]
        if(score[0] <= 0.5):
            print("CNN score")
            print(score[0])
            sentiment = "Negative"
        if(score[0] > 0.5 and score[0] < 0.6):
            print("CNN score")
            print(score[0])
            sentiment = "Neutral"
        if(score[0] >= 0.6):
            print("CNN score")
            print(score[0])
            sentiment = "Positive"
        return render_template("prediction.html", sentiment = sentiment)
    
    else:
        return render_template("formCNN.html")


@app.route("/sentiment_prediction_LSTM", methods=['GET', 'POST'])
def prediction_LSTM():
    if request.method == 'POST':
        text = request.form['sentence']
        sentiment = ''
        # Tokenize text
        x_test = pad_sequences(tokenizer.texts_to_sequences([text]), maxlen=maxLength)
        # Predict
        score = model_LSTM.predict([x_test])[0]
        if(score <= 0.5):
            print("LSTM score")
            print(score)
            sentiment = "Negative"
        elif(score >= 0.8):
            print("LSTM score")
            print(score)
            sentiment = "Positive"
        elif(score < 0.8 and score > 0.5):
            print("LSTM score")
            print(score)
            sentiment = "Neutral"
        return render_template("prediction.html", sentiment = sentiment)
    
    else:
        return render_template("form.html")


 
if __name__ == "__main__":
    app.run(debug=True)


# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 15:09:34 2019

@author: 123
"""

from flask import Flask, render_template,url_for, request
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
import pickle

classifier = pickle.load(open('model.pkl', 'rb'))
cv = pickle.load(open('cv.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('Home.html')

@app.route('/getInput',methods = ['POST'])
def getInput():
    if request.method == 'POST':
        data = request.form['Data']
        data = [data]
        vec = cv.transform(data).toarray()
        prediction = classifier.predict(vec)
        return render_template('result.html', pred = prediction)
    
if __name__ == '__main__':
    app.run()
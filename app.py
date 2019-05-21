import numpy as np
import pandas as pd
import json
from flask import Flask, abort, jsonify, request
import _pickle as pickle
import re
import string
from urllib.parse import urlparse

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from nltk.corpus import stopwords
from flask import render_template


application = app = Flask(__name__)   # AWS EBS expects variable name "application"


with open('pickle_vectorizer2.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

with open('pickle_classifier2.pkl', 'rb') as f:
    clf = pickle.load(f)

@app.route('/')
def idiots():

    return render_template('base.html', message= 'had a feeling these guys would fuck this shit up.')


@app.route('/api', methods=['POST']) 
def make_predict():

    # read in data
    lines = request.get_json(force=True)
    
    # get variables
    title = lines['title']
    body = lines['body']
    image = lines['image']  # not sure what to do with this yet

    text = title + ' ' + body

    def url_replace(string):
        return re.sub('http\S+|www.\S+', lambda match: urlparse(match.group()).hostname, string)
    text = pd.Series(text).apply(url_replace).values[0]

    table = str.maketrans(string.punctuation, ' '*len(string.punctuation))
    text = (pd.Series(text).str.lower().str.replace('\r', '').
            str.replace('/',' ').str.replace('  ',' ').str.replace('www','').
            str.replace('.com',' ').str.translate(table)).values[0]

    output = (pd.DataFrame(clf.predict_proba(vectorizer.transform([text])), columns=clf.classes_)
                .T.nlargest(5, [0]).reset_index().values).tolist()
    
    return jsonify(top_five = output)

if __name__ == '__main__':
    app.run(port = 8080, debug = True)
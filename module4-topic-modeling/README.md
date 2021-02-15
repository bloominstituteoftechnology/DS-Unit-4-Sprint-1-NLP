# NLP_Topic_Modeling-u4s1m4:
Topic_Modeling-414.ipynb: dataset is imbd keywords. It’s 40k rows and three columns: “review”, sentiment: [“positive”, “negative”], and “keywords”. We use spacy stop words to remove stop words from phrases in the “keywords” column. Result is saved as column “clean_keywords”. Each row of it represents a single document.

### Create a dictionary of phrases: 
using gensim.corpora.Dictionary() we create a dictionary of phrases across all documents. with an index number as the key and a phrase as the value.
```
id2word = corpora.Dictionary(df['clean_keywords'])
```
This results in a dictionary of 490K items. `id2word[0]` returns '1 oz episode' and `id2word.token2id['agenda']` returns 'agenda'. Next we do a statistical trimming by removing the phrases that are too frequent or very rare `id2word.filter_extremes(no_below=15, no_above=0.85)`. That reduces the size of the dictionary to 10k.

### Convert a document to a bag of words: 
The dictionary has a .doc2bow() that counts any document to bag of words. token_ids are used from the dictionary and words that are not in the dictionary are ignored `id2word.doc2bow(['agenda', 'agenda', 'city', 'aryans'])` [(0, 2), (5, 1)].
This counts number of occurrences of each token (must exist in dictionary of id2word) **per document**, (token_id, count)
```
corpus = [id2word.doc2bow(text) for text in df['clean_keywords']]
```

### Topic Modeling with Latent Dirichlet Allocation, LDA: 
The constructor estimates Latent Dirichlet Allocation model parameters based on a training corpus. `lda = LdaMulticore(corpus, id2word=id2word, num_topics=20)` You can then infer topic distributions on new or existing documents, with `doc_lda = lda[doc_bow]`. The model can be updated (trained) with new documents `lda.update(other_corpus)`. Model persistency is achieved through its load/save methods. We can see the topics by `lda.print_topics()`.
```
lda.print_topics(2, num_words=4)
[(18, '0.017*"life" + 0.015*"story" + 0.012*"love" + 0.010*"characters"'),
 (15, '0.023*"story" + 0.014*"characters" + 0.012*"people" + 0.011*"end"')]
```

To see the percentage of every topic used in a document:
```
lda[corpus[0]]
[(4, 0.3299404),
 (7, 0.11100042),
 (9, 0.051710602),
 (12, 0.3496185),
 (14, 0.1368771)]

```
The format is (topic number, percentage of doc)

### Topics Visualization:
We use pyLDAvis for topic display. 
```
vis = pyLDAvis.gensim.prepare(topic_model=lda, corpus=corpus, dictionary=id2word)
pyLDAvis.save_html(vis, "./top_20_topics.html")
```
There is a Lambda slider that can be adjusted to view different aspect of the topics.
Lambda = 0 shows the unique representative keywords of a topic, not necessarily the most repeated one
Lambda=1 shows the most probable keyword showing in a topic

### Topics in a documents:
With `lda(doc)` we make a dataframe (named topics) with rows being documents and columns being different topics and values to be the percentage of the document being of a topic. `topics.shape` is (40436, 20), for 20 topics and 40K documents. Based on that `df['primary_topic'] = topics.idxmax(axis=1)` represents the primary topic in each document. and `df['primary_topic'].value_counts()` highlights how many documents of each topic we have. Next, `agg = pd.pivot_table(data=df, values=['review'], index=['primary_topic'], columns=['sentiment'], aggfunc='count')` says how many negative and positive reviews for each primary topic is and we can plot. Finally it’s visualized by `sns.barplot(x='negative', y='primary_topic', data=agg, label="Negative", color="r", alpha=0.6)`

### Topic perplexity and topic coherence in LDA models:
In the second notebook, for 3 categories of newsgroup dataset in sklearn the same techniques are applied and `lda_multicore.log_perplexity(corpus)` as a measure of perplexity is utilized. Perplexity is a statistical measure of how well a probability model predicts a sample. For LDA models lower values of perplexity indicate lower misrepresentation of the words of the test documents by the trained topics.
Topic Coherence score measures a single topic by measuring the degree of semantic similarity between high scoring words in the topic.
```
coherence_model_lda = CoherenceModel(model=lda_multicore, texts=df['lemmas'], dictionary=id2word, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
coherence_lda
```
The higher the score the better the model is. To select the optimum number of topics we can sweep the number of topics and train different LDA models and compare their coherence scores.

### Libraries:
```
import numpy as np
import pandas as pd

import os
import re
from ast import literal_eval
import warnings
from pprint import pprint

import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from gensim import corpora
from gensim.models import CoherenceModel
from gensim import models
from gensim.models.ldamulticore import LdaMulticore

import pyLDAvis
import pyLDAvis.gensim
import tqdm
import spacy

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
```

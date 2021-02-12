# NLP-Vector_Representation-u4s1m2:
Vector_Representations-412.ipynb: There are different approaches to convert raw text to numbers for further processing.

### Bag of Words (frequency based embedding): 
This is done by representing each document as a row in a DataFrame and creating a column for each unique word in the corpora (group of documents). The presence or lack of a given word in a document is then represented either as a raw count of how many times a given word appears in a document (CountVectorizer) or as that word's Term Freq-Inverse Doc Freq score (TfidfVectorizer). The overall matrix for all documents is named Doc Term Matrix, DTM.
Here we have 401 articles from bbc website. We want to represent each document by vectors and query a document by similarity. Scikit-Learn.feature_extraction provides CountVectorizer() and TfidfVectorizer() for bag of words technique. 

#### CountVectorizer 
assigns an index to each word and counts the number of the repetition in each doc.

> vect = CountVectorizer()

> vect.fit(text)

> dtm = vect.transform(text)

> dtm = pd.DataFrame(dtm.todense(), columns=vect.get_feature_names())

    
#### TfidfVectorizer:
The purpose of TF-IDF is to find what is unique to each document. It penalizes the common terms that exist in most documents.
* **TF:** percentage of a term in a doc
* **DF:** # of docs with the term to total # of docs

> tfidf = TfidfVectorizer(stop_words='english', max_features=5000)

> dtm = tfidf.fit_transform(data)

By default it uses sklearn tokenizer. However we can use a customized tokenizer such as the one in Spacy if we wish.

* **n-gram:** The vectorizer can combine the tokens to achieve a new meaning or context.

> nlp = spacy.load("en_core_web_lg")

> tfidf = TfidfVectorizer(ngram_range=(1,2), max_df=.97, min_df=.01, tokenizer=lambda doc: [token.lemma_.strip() for token in nlp(doc) if (token.is_stop !=True) and (token.is_punct != True)]

> dtm = tfidf.fit_transform(data)

> dtm = pd.DataFrame(dtm.todense(), columns=tfidf.get_feature_names())


### Query documents by similarity: 
We will apply n-dimensional distance of two document vectors to measure similarity, and query for input and output.
* **Cosine similarity:**  It is the dot product of the two vectors divided by the product of the two vectors' lengths. Or multiplication of their magnitude along a common axis. Same vectors (duplicate documents have a cosine of 1.


> from sklearn.metrics.pairwise import cosine_similarity

> dist_matrix  = cosine_similarity(dtm)	# square matrix (#docs X #docs)

> df = pd.DataFrame(dist_matrix)

> """Grab the first 5 rows the are similar to first article excluding the same document"""

> df[df[0] < 1][0].sort_values(ascending=False)[:5]


* **NearestNeighbor (K-NN):**  To address the computational inefficiencies of the brute-force Cosine approach, tree-based data structures such as KD-Tree is used. In high dimensionality the ball tree data structure performs better. Where KD trees partition data along Cartesian axes, ball trees partition data in a series of nesting hyper-spheres. This makes tree construction more costly than that of the KD tree, but results in a data structure which can be very efficient on highly structured data, even in very high dimensions. With this setup, a single distance calculation between a test point and the centroid is sufficient to determine a lower and upper bound on the distance to all points within the node.


> from sklearn.neighbors import NearestNeighbors

> nn = NearestNeighbors(n_neighbors=5, algorithm='kd_tree')

> nn.fit(dtm)

> """returns a list of five nearest neighbors distance, and those neighbors' row numbers"""

> nn.kneighbors([dtm.iloc[0].values])

> """Query random article outside train set"""

> new_dtm = tfidf.transform(random_article)

> nn.kneighbors(new_dtm.todense())

* **Spacy similarity method:**  Spacy provides a similarity method to compare two documents:

> doc1 = nlp("It's a warm summer day")

> doc3 = nlp("There is thunderstorm in the forecast")

> sim13 = doc1.similarity(doc3)


### word embedding models to create document vectors word2vec: 
Bag of words uses stemming and lemmatization and fails to capture the context of the term or the document itself. Word2Vec like Bag-of-words learns a real-value vector representation for a predefined fixed-size vocabulary that is generated from a corpus of text, but better preserve the textual context semantic (meaning) and syntactic (usage) wise. word2vec works based on distribution hypothesis. Practically speaking, this means that if two words are found to have similar words both to the right and to the left of them throughout the corpora then those words have the same context or meaning. Each column vector represents the word's context -in this case defined by the words to the left and right of the center word. How far we look to the left and right of a given word is referred to as our "window of context". Each row vector represents the different usages of a given word, by putting the number of usage based on the text. This vectorization translates our documents from a text representation to a numeric one in a way that preserves information about the underlying context. words that have a similar context will have similar row-vector representations.



* **skip-gram:** The Skip-Gram method predicts the neighbors’ of a word given a center word. In the skip-gram model, we take a center word and a window of context (neighbors) words to train the model and then predict context words out to some window size for each center word.

* **Continuous Bag of Words:** This model takes thes opposite approach from the skip-gram model in that it tries to predict a center word based on the neighboring words. In the case of the CBOW model, we input the context words within the window (such as “the”, “Proshe”, “drove”) and aim to predict the target or center word “speedy”.
W2V can result in really large and complex vectorizations. In fact, you need Deep Neural Networks to train your Word2Vec models from scratch. W2V focuses less on document topic-modeling. The vectorizations don't really retain much information about the original document that the information came from. Spacy has a pre trained Word2Vec model.

* **Spacy W2V pre-trained model:** The model creates a 300 dimension vector for a token or even a text document.


> nlp = spacy.load('en_core_web_lg')

> doc = nlp("Two bananas in pyjamas")

> bananas_vector = doc.vector


We can reduce the dimension of the vectors from 300 to 2 and use that as coordinates to plot different texts.


> from sklearn.decomposition import PCA

> words = ['machine learning', 'artificial intelligence', 'data', 'science']

> pca = PCA(n_components=2)

> word_vecs_2d = pca.fit_transform([nlp(word).vector for word in words])

> plt.scatter(word_vecs_2d[:,0], word_vecs_2d[:,1])

We can also embed each of 401 documents and run knn on them.

> X = [nlp(str(d)).vector for d in data]

> X_df = pd.DataFrame(X)

> nn = NearestNeighbors(n_neighbors=5, algorithm='kd_tree')

> nn.fit(X_df)

> """query neighbors of document 0"""

> nn.kneighbors(X_df.iloc[0].values.reshape(1,-1))


Vector_Representations-412a.ipynb: This notebook analyzes job listing description with html tags. In addition to regular expressions , Beautiful Soup, a Python library for pulling data out of HTML and XML files is used to clean up the texts. A vectorizer function that works on both single document (to be used as tfidf vectorizer) and multiple docs (to take advantage of the pipe(batch) of nlp is defined. dtm matrix is generated and used in knn to predict document similarity.

Libraries:

```
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
import spacy
from spacy.tokenizer import Tokenizer

import os
```

# NLP-document-classification-u4s1m3:
From sklearn.datasets we use fetch_20newsgroups with two categories of “alt.atheism” and “talk.religion.misc”. We have about 900 rows with text and the target label is 0,1 specifying the category of the text. 

### Text classification:
We use a pipeline of TfidfVectorizer() and RandomForestClassifier() to classify a text query. Similar to the past for hyperparameter tuning we use GridSearchCV() on the pipeline. We can exclude the vectorizer to avoid vectorization on each iteration and save on computing time, or let them interact to capture cross tuning effect. 
* **Latent Semantic Indexing:** It refers to words that are frequently found together, as they share the same context. In sklearn we have TruncatedSVD that reduces the dimensionality based on singular value decomposition. Unlike PCA, SVD does not center the data which means it works well with sparse matrices. In particular, truncated SVD works on term count/tf-idf matrices as returned by the vectorizers in sklearn.feature_extraction.text. In that context, it is known as latent semantic analysis (LSA). Applying the LS Analysis to our pipeline:
```
vect = TfidfVectorizer(ngram_range=(1,2))
svd = TruncatedSVD(n_components=100)
lsi = Pipeline([('vect', vect), ('svd', svd)])
rfc =  RandomForestClassifier()
pipe = Pipeline([('lsi', lsi), ('clf',rfc)])
params = {'lsi__svd__n_components': [10,100,250], 'lsi__vect__max_df':[.9, .95, 1.0], 'clf__n_estimators':[5,10,20]}
search = GridSearchCV(pipe,params, cv=5, n_jobs=-1, verbose=1)
search.fit(data.data, data.target)
search.best_score_
```
We get an accuracy of 0.91.

* **Spacy Embedding**: Instead of the bag of words vectorizer we can use word2vec embedding like .vector() method for spacy docs. For this exercise there is a whiskey review dataset with description as input text and rating as multi class label. Here we also have a test set to predict.
```
X = get_word_vectors(train['description'])
rfc.fit(X, train['category'])
test['category'] = rfc.predict(X_test)
test[['id', 'category']].to_csv('testSolutionSubmission.csv', header=True, index=False)
```
To save on time we skip putting the embedding process in a hyper-tune search.

### Libraries:
```
from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import TruncatedSVD
from sklearn import set_config
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
import pandas as pd
```

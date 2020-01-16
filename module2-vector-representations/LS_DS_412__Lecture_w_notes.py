#!/usr/bin/env python
# coding: utf-8

# Lambda School Data Science
# 
# *Unit 4, Sprint 1, Module 2*
# 
# ---
# 
# # Vector Representations (Prepare)
# 
# 
# As we learned yesterday, machines cannot intrepret raw text. We need to transform that text into something we/machines can more readily analyze. Yesterday, we did simple counts of counts to summarize the content of Amazon reviews. Today, we'll extend those concepts to talk about vector representations such as Bag of Words (BoW) and word embedding models. We'll use those representations for search, visualization, and prepare for our classification day tomorrow. 
# 
# Processing text data to prepare it for maching learning models often means translating the information from documents into a numerical format. Bag-of-Words approaches (sometimes referred to as Frequency-Based word embeddings) accomplish this by "vectorizing" tokenized documents. This is done by representing each document as a row in a dataframe and creating a column for each unique word in the corpora (group of documents). The presence or lack of a given word in a document is then represented either as a raw count of how many times a given word appears in a document (CountVectorizer) or as that word's TF-IDF score (TfidfVectorizer).
# 
# On the python side, we will be focusing on `sklearn` and `spacy` today.  
# 
# ## Case Study
# 
# We're going to pretend we're on the datascience team at the BBC. We want to recommend articles to visiters to on the BBC website based on the article they just read. Our team wants 
# 
# **Dataset:**
# 
# [D. Greene and P. Cunningham. "Practical Solutions to the Problem of Diagonal Dominance in Kernel Document Clustering", Proc. ICML 2006.](http://mlg.ucd.ie/datasets/bbc.html)
# *Please note that the dataset has been sampled down to tech articles only.* 
# 
# ## Learning Objectives
# * <a href="#p1">Part 1</a>: Represent a document as a vector
# * <a href="#p2">Part 2</a>: Query Documents by Similarity
# * <a href="#p3">Part 3</a>: Apply word embedding models to create document vectors

# # Represent a document as a vector (Learn)
# <a id="p1"></a>

# ## Overview
# 
# In this section, we are going to create Document Term Matrices (DTM). Each column represents a word. Each row represents a document. The value in each cell can be range of different things. The most traditional: counts of appearences of words, does the word appear at all (binary), and term-frequency inverse-document frequence (TF-IDF). 
# 
# **Discussion:** Don't we loose all the context and grammer if we do this? So Why does it work?

# ## Follow Along

# In[22]:


""" Import Statements """

# Classics
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA

import spacy
nlp = spacy.load("en_core_web_lg")


# **Warm Up (_3 Minutes_)**
# 
# Extract the tokens from this sentence using Spacy. Text is from [OpenAI](https://openai.com/blog/better-language-models/)

# In[23]:


text = "We created a new dataset which emphasizes diversity of content, by scraping content from the Internet. In order to preserve document quality, we used only pages which have been curated/filtered by humans—specifically, we used outbound links from Reddit which received at least 3 karma. This can be thought of as a heuristic indicator for whether other users found the link interesting (whether educational or funny), leading to higher data quality than other similar datasets, such as CommonCrawl."


# In[24]:


doc = nlp(text)

print([token.lemma_ for token in doc if (token.is_stop != True) and (token.is_punct != True)])


# In[25]:


list(doc.noun_chunks)


# In[26]:


import os 
def gather_data(filefolder):
    """ Produces List of Documents from a Directory
    
    filefolder (str): a path of .txt files
    
    returns list of strings 
    """
    
    data = []
    
    files = os.listdir(filefolder)
    
    for article in files: 
        
        path = os.path.join(filefolder, article)
                    
        if  path[-3:] == 'txt':
            with open(path, 'rb') as f:
                data.append(f.read())
    
    return data


# In[27]:


data = gather_data('./data')


# In[28]:


data[0]


# ### CountVectorizer

# In[29]:


from sklearn.feature_extraction.text import CountVectorizer

# list of text documents
text = ["We created a new dataset which emphasizes diversity of content, by scraping content from the Internet."," In order to preserve document quality, we used only pages which have been curated/filtered by humans—specifically, we used outbound links from Reddit which received at least 3 karma."," This can be thought of as a heuristic indicator for whether other users found the link interesting (whether educational or funny), leading to higher data quality than other similar datasets, such as CommonCrawl."]

# create the transformer
vect = CountVectorizer()

# tokenize and build vocab
vect.fit(text)

# transform text
dtm = vect.transform(text)

# Create a Vocabulary
# The vocabulary establishes all of the possible words that we might use.

# The vocabulary dictionary does not represent the counts of words!!


# In[30]:


print(vect.get_feature_names())


# In[32]:


print(dtm)


# In[33]:


text[:25]


# In[36]:


# Dealing with Sparse Matrix
# pandas can not read a sparse matrix

dtm.todense()


# In[38]:


# Get Word Counts for each document
dtm = pd.DataFrame(dtm.todense(), columns=vect.get_feature_names())
dtm


# In[39]:


data[0][:25]


# In[40]:


len(data)


# In[16]:


data[0]


# **Three Minute Challenge:** 
# * Apply CountVectorizer to our BBC Data
# * Store results in a dataframe called `dtm`
# * Extra Challenge - Try to Customize CountVectorizer with Spacy Processing

# In[42]:


# Apply CountVectorizer to our Data
# Use custom Spacy Vectorizer

vect = CountVectorizer(stop_words='english')

# learn vocab
vect.fit(data)

# get sparse dtm
dtm = vect.transform(data)
dtm = pd.DataFrame(dtm.todense(), columns=vect.get_feature_names())

# BBC articles in `data` variable 


# In[43]:


dtm.head()


# In[44]:


doc_len = [len(doc) for doc in data]


# In[46]:


import seaborn as sns

sns.distplot(doc_len, color='pink');


# ### TfidfVectorizer
# 
# ## Term Frequency - Inverse Document Frequency (TF-IDF)
# 
# <center><img src="https://mungingdata.files.wordpress.com/2017/11/equation.png?w=430&h=336" width="300"></center>
# 
# Term Frequency: Percentage of words in document for each word
# 
# Document Frequency: A penalty for the word existing in a high number of documents.
# 
# The purpose of TF-IDF is to find what is **unique** to each document. Because of this we will penalize the term frequencies of words that are common across all documents which will allow for each document's most different topics to rise to the top.

# In[48]:


from sklearn.feature_extraction.text import TfidfVectorizer

# Instantiate vectorizer object
tfidf = TfidfVectorizer(stop_words='english')

# Create a vocabulary and get word counts per document
dtm = tfidf.fit_transform(data)

# Print word counts

# Get feature names to use as dataframe column headers
dtm = pd.DataFrame(dtm.todense(), columns=tfidf.get_feature_names())

# View Feature Matrix as DataFrame
dtm.head()


# In[52]:


def tokenize(document):
    
    doc = nlp(document)
    
    return [token.lemma_.strip() for token in doc if (token.is_stop != True) and (token.is_punct != True)]


# In[53]:


# Tunning Parameters
# instantiate vec object
tfidf = TfidfVectorizer(tokenizer=tokenize, min_df=0.025, max_df=.98, ngram_range=(1,2))

# create vocab and get word count
# similar to fit_predict
dtm = tfidf.fit_transform(data)

# get feature names to use as dataframe col headers
dtm = pd.DataFrame(dtm.todense(), columns=tfidf.get_feature_names())

# view dtm
dtm.head()


# In[54]:


dtm.shape


# ## Challenge
# 
# During this module's project assignment, you will transform data science job listings to vector representations for analysis downstream. 

# In[ ]:





# # Query Documents by Similarity (Learn)
# <a id="p2"></a>

# ## Overview
# 
# Have you ever considered how a search bar works? You may just think that search bars simply match your input text againist the documents. While there are many different mechanisms for the 'match', one of the most classic is to search by similarity. We will apply n-dimensional distance to measure similarity, and query for input and output. 

# ## Follow Along

# ### Cosine Similarity (Brute Force)

# In[55]:


# Calculate Distance of TF-IDF Vectors
from sklearn.metrics.pairwise import cosine_similarity

dist_matrix  = cosine_similarity(dtm)


# In[56]:


# Turn it into a DataFrame

df = pd.DataFrame(dist_matrix)


# In[57]:


# Our Similarity Matrix is ? size 
df.shape


# In[58]:


# Each row is the similarity of one document to all other documents (including itself)
df[0][:5]


# In[60]:


# Grab the row
df[0].sort_values(ascending=False)[2:7]


# In[61]:


print(data[0][:150])


# In[62]:


print(data[297][:180])


# ### NearestNeighbor (K-NN) 
# 
# To address the computational inefficiencies of the brute-force approach, a variety of tree-based data structures have been invented. In general, these structures attempt to reduce the required number of distance calculations by efficiently encoding aggregate distance information for the sample. The basic idea is that if point  is very distant from point , and point  is very close to point , then we know that points  and  are very distant, without having to explicitly calculate their distance. In this way, the computational cost of a nearest neighbors search can be reduced to  or better. This is a significant improvement over brute-force for large data.
# 
# To address the inefficiencies of KD Trees in higher dimensions, the ball tree data structure was developed. Where KD trees partition data along Cartesian axes, ball trees partition data in a series of nesting hyper-spheres. This makes tree construction more costly than that of the KD tree, but results in a data structure which can be very efficient on highly structured data, even in very high dimensions.
# 
# A ball tree recursively divides the data into nodes defined by a centroid  and radius , such that each point in the node lies within the hyper-sphere defined by  and . The number of candidate points for a neighbor search is reduced through use of the triangle inequality:
# 
# With this setup, a single distance calculation between a test point and the centroid is sufficient to determine a lower and upper bound on the distance to all points within the node. Because of the spherical geometry of the ball tree nodes, it can out-perform a KD-tree in high dimensions, though the actual performance is highly dependent on the structure of the training data. In scikit-learn, ball-tree-based neighbors searches are specified using the keyword algorithm = 'ball_tree', and are computed using the class sklearn.neighbors.BallTree. Alternatively, the user can work with the BallTree class directly.

# In[63]:


dtm.head()


# In[65]:


# Instantiate
from sklearn.neighbors import NearestNeighbors

# Fit on TF-IDF Vectors
nn  = NearestNeighbors(n_neighbors=5, algorithm='kd_tree')
nn.fit(dtm)


# In[66]:


# Query Using kneighbors 
nn.kneighbors([dtm.iloc[0]])


# In[ ]:





# In[67]:


medium_random_ass_tech = [ """1. The VR and AI Enhancement
The list of trends in VR can never be complete without Artificial Intelligence. The duo has the potential to change the world. The two are in their infancy, to begin with, but they have made some appearances.

Take the example of Instagram and Snapchat. The dog faces and other funny filters are a creation of AI and VR. If you are looking for a scientific example, the Machine Learning Microscope of Google is a perfect example. This tool is capable of highlighting cancerous tissues. Now we are just waiting for AR and VR to create something powerful.

2. Consumer Entertainment
Oculus has partnered with the NBA. The brand is using VR to provide state of the art entertainment to its users. Yes, you have guessed it right. While you are wearing the Oculus VR headsets sitting on the courtside, you will feel as if you are actually in the NBA game. Cool, right?

VR headset technology is becoming accessible than before. It’s not just limited to gamers. TV watchers can get entertainment as well.

3. Education and Training
Education and training are one of the most costly and dangerous exercises in the workplace. With VR technology, organizations can cut back on the costs and give their employees the best training without exposing them to risk.

Recently, Walmart has used 17,000 Oculus Go headsets for training its employees in the customer service department. Similarly, the U.S. Army has been using Microsoft HoloLens technology to offer soldiers real-time updates on their environment.

4. Travel and Tourism
Traveling through VR is a different kind of traveling. The immersive video allows you to experience your destinations before even packing your suitcase. With VR, you can have the “try it before buy it experience.”

If you think you will feel claustrophobic in the cruise stateroom, then why not test the waters? Maybe you will feel better in a suit. VR allows you to explore the cruise ship rooms as well as the rooms in a hotel prior to booking it.

Apart from checking your room, you can also have a street view, check the venues, and restaurants nearby. With this virtual tour, you can decide whether the destination you have picked for yourself is good or not.

That’s not all, with VR travel apps, you can even relive your vacation after returning home. Now that’s something you will cherish for sure.

What’s the Future of VR?
Experts say the future of VR is location-based. But wait, it has nothing to do with the GPS technology. Location-based means bringing the VR experience to users where they are regardless of their location. They will be introduced to technology without having to purchase it.

However, for that, customers need to be aware of the proactiveness of VR. They must have an interest in VR for home as well. It still has a long way to go.

Conclusion
Consumers in today’s time are looking for experience. VR has the ability to play a huge role in the future of learning and development. With virtual simulation, a user can have real life-like experience at a fraction of cost.

The future for Virtual Reality is still in progress. It’s pretty much like the Cox channel lineup for different channels. Not all popular channels are available in all areas. Some of the VR tech and apps are in use but a majority of them are still in progress."""]


# In[68]:


# Query for Sim of Random doc to BBC
new = tfidf.transform(medium_random_ass_tech)


# In[69]:


new


# In[70]:


nn.kneighbors(new.todense())


# In[71]:


# Inspect Most relevant result
data[255]


# ## Challenge
# 
# In the module project assignment, you will apply one of these search techniques to retrieve documents related to a query document. 

# # Apply word embedding models to create document vectors (Learn)
# <a id="p3"></a>

# ## Overview
# ### BoW discards textual context
# 
# One of the limitations of Bag-of-Words approaches is that any information about the textual context surrounding that word is lost. This also means that with bag-of-words approaches often the only tools that we have for identifying words with similar usage or meaning and subsequently consolidating them into a single vector is through the processes of stemming and lemmatization which tend to be quite limited at consolidating words unless the two words are very close in their spelling or in their root parts-of-speech.
# 
# ### Embedding approaches preserve more textual context
# Word2Vec is an increasingly popular word embedding technique. Like Bag-of-words it learns a real-value vector representation for a predefined fixed-size vocabulary that is generated from a corpus of text. However, in contrast to BoW, Word2Vec approaches are much more capable of accounting for textual context, and are better at discovering words with similar meanings or usages (semantic or syntactic similarity).
# 
# ### Word2Vec Intuition
# ### The Distribution Hypothesis
# 
# In order to understand how Word2Vec preserves textual context we have to understand what's called the Distribution Hypothesis (Reference: Distribution Hypothesis Theory  -https://en.wikipedia.org/wiki/Distributional_semantics. The Distribution Hypothesis operates under the assumption that words that have similar contexts will have similar meanings. Practically speaking, this means that if two words are found to have similar words both to the right and to the left of them throughout the corpora then those words have the same context and are assumed to have the same meaning. 
# 
# > "You shall know a word by the company it keeps" - John Firth
# 
# This means that we let the usage of a word define its meaning and its "similarity" to other words. In the following example, which words would you say have a similar meaning? 
# 
# **Sentence 1**: Traffic was light today
# 
# **Sentence 2**: Traffic was heavy yesterday
# 
# **Sentence 3**: Prediction is that traffic will be smooth-flowing tomorrow since it is a national holiday
# 
# What words in the above sentences seem to have a similar meaning if all you knew about them was the context in which they appeared above? 
# 
# Lets take a look at how this might work in action, the following example is simplified, but will give you an idea of the intuition for how this works.
# 
# #### Corpora:
# 
# 1) "It was the sunniest of days."
# 
# 2) "It was the raniest of days."
# 
# #### Vocabulary:
# 
# {"it": 1, "was": 2, "the": 3, "of": 4, "days": 5, "sunniest": 6, "raniest": 7}
# 
# ### Vectorization
# 
# |       doc   | START_was | it_the | was_sunniest | the_of | sunniest_days | of_it | days_was | it_the | was_raniest | raniest_days | of_END |
# |----------|-----------|--------|--------------|--------|---------------|-------|----------|--------|-------------|--------------|--------|
# | it       | 1         | 0      | 0            | 0      | 0             | 0     | 1        | 0      | 0           | 0            | 0      |
# | was      | 0         | 1      | 0            | 0      | 0             | 0     | 0        | 1      | 0           | 0            | 0      |
# | the      | 0         | 0      | 1            | 0      | 0             | 0     | 0        | 0      | 1           | 0            | 0      |
# | sunniest | 0         | 0      | 0            | 1      | 0             | 0     | 0        | 0      | 0           | 0            | 0      |
# | of       | 0         | 0      | 0            | 0      | 1             | 0     | 0        | 0      | 0           | 1            | 0      |
# | days     | 0         | 0      | 0            | 0      | 0             | 0     | 0        | 0      | 0           | 0            | 1      |
# | raniest  | 0         | 0      | 0            | 1      | 0             | 0     | 0        | 0      | 0           | 0            | 0      |
# 
# Each column vector represents the word's context -in this case defined by the words to the left and right of the center word. How far we look to the left and right of a given word is referred to as our "window of context." Each row vector represents the the different usages of a given word. Word2Vec can consider a larger context than only words that are immediately to the left and right of a given word, but we're going to keep our window of context small for this example. What's most important is that this vectorization has translated our documents from a text representation to a numeric one in a way that preserves information about the underlying context. 
# 
# We can see that words that have a similar context will have similar row-vector representations, but before looking that more in-depth, lets simplify our vectorization slightly. You'll notice that we're repeating the column-vector "it_the" twice. Lets combine those into a single vector by adding them element-wise. 
# 
# |       *   | START_was | it_the | was_sunniest | the_of | sunniest_days | of_it | days_was | was_raniest | raniest_days | of_END |
# |----------|-----------|--------|--------------|--------|---------------|-------|----------|-------------|--------------|--------|
# | it       | 1         | 0      | 0            | 0      | 0             | 0     | 1        | 0           | 0            | 0      |
# | was      | 0         | 2      | 0            | 0      | 0             | 0     | 0        | 0           | 0            | 0      |
# | the      | 0         | 0      | 1            | 0      | 0             | 0     | 0        | 1           | 0            | 0      |
# | sunniest | 0         | 0      | 0            | 1      | 0             | 0     | 0        | 0           | 0            | 0      |
# | of       | 0         | 0      | 0            | 0      | 1             | 0     | 0        | 0           | 1            | 0      |
# | days     | 0         | 0      | 0            | 0      | 0             | 0     | 0        | 0           | 0            | 1      |
# | raniest  | 0         | 0      | 0            | 1      | 0             | 0     | 0        | 0           | 0            | 0      |
# 
# Now, can you spot which words have a similar row-vector representation? Hint: Look for values that are repeated in a given column. Each column represents the context that word was found in. If there are multiple words that share a context then those words are understood to have a closer meaning with each other than with other words in the text.
# 
# Lets look specifically at the words sunniest and raniest. You'll notice that these two words have exactly the same 10-dimensional vector representation. Based on this very small corpora of text we would conclude that these two words have the same meaning because they share the same usage. Is this a good assumption? Well, they are both referring to the weather outside so that's better than nothing. You could imagine that as our corpora grows larger we will be exposed a greater number of contexts and the Distribution Hypothesis assumption will improve. 
# 
# ### Word2Vec Variants
# 
# #### Skip-Gram
# 
# The Skip-Gram method predicts the neighbors’ of a word given a center word. In the skip-gram model, we take a center word and a window of context (neighbors) words to train the model and then predict context words out to some window size for each center word.
# 
# This notion of “context” or “neighboring” words is best described by considering a center word and a window of words around it. 
# 
# For example, if we consider the sentence **“The speedy Porsche drove past the elegant Rolls-Royce”** and a window size of 2, we’d have the following pairs for the skip-gram model:
# 
# **Text:**
# **The**	speedy	Porsche	drove	past	the	elegant	Rolls-Royce
# 
# *Training Sample with window of 2*: (the, speedy), (the, Porsche)
# 
# **Text:**
# The	**speedy**	Porsche	drove	past	the	elegant	Rolls-Royce
# 
# *Training Sample with window of 2*: (speedy, the), (speedy, Porsche), (speedy, drove)
# 
# **Text:**
# The	speedy	**Porsche**	drove	past	the	elegant	Rolls-Royce
# 
# *Training Sample with window of 2*: (Porsche, the), (Porsche, speedy), (Porsche, drove), (Porsche, past)
# 
# **Text:**
# The	speedy	Porsche	**drove**	past	the	elegant	Rolls-Royce
# 
# *Training Sample with window of 2*: (drove, speedy), (drove, Porsche), (drove, past), (drove, the)
# 
# The **Skip-gram model** is going to output a probability distribution i.e. the probability of a word appearing in context given a center word and we are going to select the vector representation that maximizes the probability.
# 
# With CountVectorizer and TF-IDF the best we could do for context was to look at common bi-grams and tri-grams (n-grams). Well, skip-grams go far beyond that and give our model much stronger contextual information.
# 
# ![alt text](https://www.dropbox.com/s/c7mwy6dk9k99bgh/Image%202%20-%20SkipGrams.jpg?raw=1)
# 
# ## Continuous Bag of Words
# 
# This model takes thes opposite approach from the skip-gram model in that it tries to predict a center word based on the neighboring words. In the case of the CBOW model, we input the context words within the window (such as “the”, “Proshe”, “drove”) and aim to predict the target or center word “speedy” (the input to the prediction pipeline is reversed as compared to the SkipGram model).
# 
# A graphical depiction of the input to output prediction pipeline for both variants of the Word2vec model is attached. The graphical depiction will help crystallize the difference between SkipGrams and Continuous Bag of Words.
# 
# ![alt text](https://www.dropbox.com/s/k3ddmbtd52wq2li/Image%203%20-%20CBOW%20Model.jpg?raw=1)
# 
# ## Notable Differences between Word Embedding methods:
# 
# 1) W2V focuses less document topic-modeling. You'll notice that the vectorizations don't really retain much information about the original document that the information came from. At least not in our examples.
# 
# 2) W2V can result in really large and complex vectorizations. In fact, you need Deep Neural Networks to train your Word2Vec models from scratch, but we can use helpful pretrained embeddings (thank you Google) to do really cool things!
# 
# *^ All that noise....AND Spacy has pretrained a Word2Vec model you can just use? WTF JC?*
# 
# Let's take a look at how to do it. 

# In[73]:


# Process a text
doc = nlp("Two bananas in pyjamas")

# Get the vector for the token "bananas"
bananas_vector = doc.vector
print(bananas_vector)


# In[74]:


len(bananas_vector)


# In[75]:


doc1 = nlp("It's a warm summer day")
doc2 = nlp("It's sunny outside")

# Get the similarity of doc1 and doc2
similarity = doc1.similarity(doc2)
print(similarity)


# In[76]:


# import the PCA module from sklearn
from sklearn.decomposition import PCA

def get_word_vectors(words):
    # converts a list of words into their word vectors
    return [nlp(word).vector for word in words]

words = ['car', 'truck', 'suv', 'race', 'elves', 'dragon', 'sword', 'king', 'queen', 'prince', 'horse', 'fish' , 'lion', 'tiger', 'lynx', 'potato']

# intialise pca model and tell it to project data down onto 2 dimensions
pca = PCA(n_components=2)

# fit the pca model to our 300D data, this will work out which is the best 
# way to project the data down that will best maintain the relative distances 
# between data points. It will store these intructioons on how to transform the data.
pca.fit(get_word_vectors(words))

# Tell our (fitted) pca model to transform our 300D data down onto 2D using the 
# instructions it learnt during the fit phase.
word_vecs_2d = pca.transform(get_word_vectors(words))

# let's look at our new 2D word vectors
word_vecs_2d


# In[77]:


# create a nice big plot 
plt.figure(figsize=(20,15))

# plot the scatter plot of where the words will be
plt.scatter(word_vecs_2d[:,0], word_vecs_2d[:,1])

# for each word and coordinate pair: draw the text on the plot
for word, coord in zip(words, word_vecs_2d):
    x, y = coord
    plt.text(x, y, word, size= 15)

# show the plot
plt.show()


# ## Follow Along
# ### Extract Document Vectors
# 
# Let's see how much the quality of our query will work when we try a new embedding model.
# 
# Steps:
# * Extract Vectors from Each Document
# * Search using KNN
# 

# In[78]:


X = [nlp(str(d)).vector for d in data]


# In[79]:


X[0]


# ## Challenge
# 
# You will extract word embeddings from documents using Spacy's pretrained model in the upcoming module project. 

# # Review
# For your module project assignment you will create vector repsentations of indeed.com Data Science job listings. You will then estimate a similarity model to perform searches for job descriptions. Get started with your [module project here](./LS_DS_412_Vector_Representations_Assignment.ipynb)

# # Sources
# 
# * Spacy 101 - https://course.spacy.io
# * NLTK Book - https://www.nltk.org/book/
# * An Introduction to Information Retrieval - https://nlp.stanford.edu/IR-book/pdf/irbookonlinereading.pdf

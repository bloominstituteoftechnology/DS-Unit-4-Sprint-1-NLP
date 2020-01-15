#!/usr/bin/env python
# coding: utf-8

# <img align="left" src="https://lever-client-logos.s3.amazonaws.com/864372b1-534c-480e-acd5-9711f850815c-1524247202159.png" width=200>
# <br></br>
# <br></br>
# 
# # Natural Language Processing (NLP)
# ## *Data Science Unit 4 Sprint 1 Assignment 1*
# 
# Your goal in this assignment: find the attributes of the best & worst coffee shops in the dataset. The text is fairly raw: dates in the review, extra words in the `star_rating` column, etc. You'll probably want to clean that stuff up for a better analysis. 
# 
# Analyze the corpus of text using text visualizations of token frequency. Try cleaning the data as much as possible. Try the following techniques: 
# - Lemmatization
# - Custom stopword removal
# 
# Keep in mind the attributes of good tokens. Once you have a solid baseline, layer in the star rating in your visualization(s). Key part of this assignment - produce a write-up of the attributes of the best and worst coffee shops. Based on your analysis, what makes the best the best and the worst the worst. Use graphs and numbesr from your analysis to support your conclusions. There should be plenty of markdown cells! :coffee:

# In[5]:


from IPython.display import YouTubeVideo

YouTubeVideo('Jml7NVYm8cs')


# In[2]:


get_ipython().run_line_magic('pwd', '')


# In[1]:


import pandas as pd

url = "https://raw.githubusercontent.com/LambdaSchool/DS-Unit-4-Sprint-1-NLP/master/module1-text-data/data/yelp_coffeeshop_review_data.csv"

shops = pd.read_csv(url)
shops.head()


# In[2]:


# Start here 


# ## How do we want to analyze these coffee shop tokens? 
# 
# - Overall Word / Token Count
# - View Counts by Rating 
# - *Hint:* a 'bad' coffee shops has a rating betweeen 1 & 3 based on the distribution of ratings. A 'good' coffee shop is a 4 or 5. 

# In[33]:


# imports
from collections import Counter
import re
import urllib.request 

# plotting
import squarify
import matplotlib.pyplot as plt
import seaborn as sns

#nlp libs
import spacy
from spacy.tokenizer import Tokenizer
from nltk.stem import PorterStemmer

nlp = spacy.load("en_core_web_lg")
tokenizer = Tokenizer(nlp.vocab)


# In[28]:


print(shops.shape)
shops.dtypes


# In[12]:


shops.head(10)


# In[26]:


shops['star_rating'].unique()


# In[29]:


# drop star_rating + convert to int
shops['star_rating'] = shops['star_rating'].str.replace('.0 star rating', '').astype('int')


# In[35]:


shops.head()


# In[18]:


shops.iloc[7]


# In[138]:


# using two samples to try to tokenize
# sample_text = shops['full_review_text'].iloc[7]
sample_text = "11/2/2016 Love this place!  5 stars for cleanliness 5 stars for fun ambiance"


# In[139]:


sample_rating = 5


# In[42]:



def tokenize(text):
    
    tokens = re.sub(r'[^a-zA-Z]', ' ', text)
    tokens = tokens.lower().split()

    return tokens


# In[43]:


tokenize(sample_text)


# In[44]:


# applying python to tokenize the full_review_text 
# + create a new column called tokens

shops['tokens'] = shops['full_review_text'].apply(tokenize)


# In[45]:


shops['tokens'].head()


# In[46]:


# new df w tokens

shops.head()


# In[67]:


# check types
shops.dtypes


# In[68]:


# using spacy tokenizer
# update tokens w/o stopwords and punctuations

tokens = []

for doc in tokenizer.pipe(shops['full_review_text'], batch_size=500):
    
    doc_tokens = []
    
    for token in doc:
        if (token.is_stop == False) & (token.is_punct == False):
            doc_tokens.append(token.text.lower())
    tokens.append(doc_tokens)
    
shops['tokens'] = tokens


# In[69]:


shops['tokens'].head()


# In[79]:


# defining count function to define the number of words in the text

def count(docs):
    
    word_count = Counter()
    appears_in = Counter()
    
    total_docs = len(docs)
    
    for doc in docs:
        word_count.update(doc)
        appears_in.update(set(doc))
        
    temp = zip(word_count.keys(), word_count.values())
    
    wc = pd.DataFrame(temp, columns = ['word', 'count'])
    
    wc['rank'] = wc['count'].rank(method='first', ascending=False)
    total = wc['count'].sum()
    
    wc['pct_total'] = wc['count'].apply(lambda x: x / total)
    
    wc = wc.sort_values(by='rank')
    wc['cul_pct_total'] = wc['pct_total'].cumsum()
    
    t2 = zip(appears_in.keys(), appears_in.values())
    ac = pd.DataFrame(t2, columns=['word', 'appears_in'])
    wc = ac.merge(wc, on='word')
    
    wc['appears_in_pct'] = wc['appears_in'].apply(lambda x: x / total_docs)
    
    return wc.sort_values(by='rank')


# In[120]:


# defin stop words list

STOP_WORDS = nlp.Defaults.stop_words.union(["check-in", " ", "it's", "1", "-", "i'm", "i've", "place", 
                                            "coffee", "got", "ordered", "the", "a", "for", "my", "austin", "be"])


# In[121]:


# using stop words list to update tokens 
# excluding common words w none or less value
""" Update tokens w/o defined stopwords and punctuation"""
tokens = []

for doc in tokenizer.pipe(shops['full_review_text'], batch_size=500):
    
    doc_tokens = []
    
    for token in doc:
        if token.text.lower() not in STOP_WORDS:
            doc_tokens.append(token.text.lower())
            
    tokens.append(doc_tokens)    

shops['tokens'] = tokens    


# In[122]:


wc = count(shops['tokens'])

wc.head()


# In[123]:


from collections import Counter

# instantiating an empty object
word_count = Counter()

# Update it based on a split
shops['tokens'].apply(lambda x: word_count.update(x))

# top 10 most common words
word_count.most_common(10)


# ## Can visualize the words with the greatest difference in counts between 'good' & 'bad'?
# 
# Couple Notes: 
# - Rel. freq. instead of absolute counts b/c of different numbers of reviews
# - Only look at the top 5-10 words with the greatest differences
# 

# In[125]:


wc_top10 = wc[wc['rank'] <= 10]

squarify.plot(sizes=wc_top10['pct_total'], label=wc_top10['word'], alpha=.9 )
plt.axis('off')
plt.show()


# In[126]:


# defining worst and best shops

worst_shops = shops[shops['star_rating'] <= 3]
best_shops = shops[shops['star_rating'] >= 4]


# In[127]:


# counting words for worst shops
tokens_worst_shops = []

""" Update tokens w/o defined stopwords and punctuation"""
for doc in tokenizer.pipe(worst_shops['full_review_text'], batch_size=500):
    
    doc_tokens_worst_shops = []
    
    for token in doc:
        if token.text.lower() not in STOP_WORDS:
            doc_tokens_worst_shops.append(token.text.lower())

    tokens_worst_shops.append(doc_tokens_worst_shops)

worst_shops['tokens_worst_shops'] = tokens_worst_shops   


# In[128]:


wc_worst_shops = count(worst_shops['tokens_worst_shops'])
wc_worst_shops.head()


# In[130]:


# defining top 10 words most commonly used for worst_shops

wc_worst_shops_top10 = wc_worst_shops[wc_worst_shops['rank'] <= 10]

squarify.plot(sizes=wc_worst_shops_top10['pct_total'], label=wc_worst_shops_top10['word'], alpha=.9)
plt.axis('off')
plt.show()


# In[131]:


# counting words for best shop

tokens_best_shops = []

for doc in tokenizer.pipe(best_shops['full_review_text'], batch_size=500):
    
    doc_tokens_best_shops = []
    
    for token in doc:
        if token.text.lower() not in STOP_WORDS:
            doc_tokens_best_shops.append(token.text.lower())
         
    tokens_best_shops.append(doc_tokens_best_shops)
    
best_shops['tokens_best_shops'] = tokens_best_shops    


# In[132]:


wc_best_shops = count(best_shops['tokens_best_shops'])
wc_best_shops.head()


# In[133]:


# defining top 10 words most commonly used for best shops

wc_best_shops_top10 = wc_best_shops[wc_best_shops['rank'] <= 10]

squarify.plot(sizes=wc_best_shops_top10['pct_total'], label=wc_best_shops_top10['word'], alpha=.9)
plt.axis('off')
plt.show()


# In[134]:


# ratings count

shops['star_rating'].value_counts()


# In[136]:


# cumulative distribution plot

sns.lineplot(x='rank', y='cul_pct_total', data=wc, color='pink');


# In[137]:


# excluding the extremes

wc = wc[wc['appears_in_pct'] >= 0.025]

sns.distplot(wc['appears_in_pct']);


# In[140]:


# applying lemma 
# instantiate base model

sample_text

nlp = spacy.load("en_core_web_lg")

doc = nlp(sample_text)

# lemma attributes
for token in doc:
    print(token.text, " ", token.lemma_)


# In[ ]:


# defining lemma function

def get_lemmas(text):
    
    lemmas = []
    
    doc = nlp(text)
    
    for token in doc:
        if ((token.is_stop == False) and (token.is_punct == False)) and (token.pos_!= '-PRON-') and (token.is_space == False) and (token.is_digit == False):
            lemmas.append(token.lemma_)
          
    return lemmas    


# In[ ]:


shops['lemmas'] = shops['full_review_text'].apply(get_lemmas)


# In[ ]:


shops['lemmas'].head()


# In[ ]:





# In[ ]:





# In[ ]:





# ## Stretch Goals
# 
# * Analyze another corpus of documents - such as Indeed.com job listings ;).
# * Play with the Spacy API to
#  - Extract Named Entities
#  - Extracting 'noun chunks'
#  - Attempt Document Classification with just Spacy
#  - *Note:* This [course](https://course.spacy.io/) will be of interesting in helping you with these stretch goals. 
# * Try to build a plotly dash app with your text data 
# 
# 

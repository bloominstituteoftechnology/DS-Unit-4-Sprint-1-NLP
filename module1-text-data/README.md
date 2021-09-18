## NLP-TextData-u4s1m1:

In [nlp_TextData-411.ipynb](https://github.com/skhabiri/ML-NLP/blob/main/module1-text-data/nlp_TextData-411.ipynb) we use [Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products.zip](https://github.com/skhabiri/ML-NLP/blob/main/module1-text-data/data/Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products_May19.csv.zip) dataset to tokenize and visualize `reviews.text` column. 
- **Tokenize by regex and str methods:** We use regex and text methods to clean up the data and tokenize each row of document into an iterable list. In order to apply token frequency method we use Counter class from collections library. We count repetition of tokens in the entire corpus of documents (data frame) as well as appearance of the tokens in each document (row). We get the ‘'pct_total' of each token and after ranking, we derive its cumulative sum. seaborn.lineplot() can be used to plot the 'cul_pct_total' vs ‘word’ or ‘rank’. Furthermore we use `squarify.plot(sizes=wc_topi['pct_total'], label=wc_topi['word'], alpha=.8 )` to viualize the frequency of the top rank tokens.
- **Tokenize by Spacy:** We can use spacy to get an iterable tokens class object.
```
from spacy.tokenizer import Tokenizer
nlp = spacy.load("en_core_web_lg")
tokenizer = Tokenizer(nlp.vocab)
[token.text for token in tokenizer(sample_text)]
# alternatively:
[token.text for token in nlp(sample_text)]
```
To manage the memory usage we pipeline the text into batches: `tokenizer.pipe(df['reviews.text'], batch_size=500)`
- **Filter out tokens by Spacy methods:** Spacy default stop words  is a python set nlp.Defaults.stop_words. the token class (not tokens class) provide token.is_stop, token.is_punct, token.text attributes to filter out default stop words, punctuations, apply string methods such as .lower() to the tokenized text. the default stop word set can be expanded by: 
```
STOP_WORDS = nlp.Defaults.stop_words.union(['it', "it's", 'it.', 'the'])
if token.text.lower() not in STOP_WORDS:
```
We can also filter out pronounce with `token.pos_ != 'PRON'`
- **Statistical trimming:** `df_wc['appears_in_pct'].describe()` can be used to find out tokens that only appear (count only once) in a few documents as well as tokens that appear in almost all the documents. Both types fail to provide insight. we can filter out similar to `wc = wc[wc['appears_in_pct'] >= 0.025]`
- **Stemming:** Is fast and useful for search engines but limited use for human as it only chops off the ing or es at the end of the words. 
```
from nltk.stem import PorterStemmer
ps = PorterStemmer()
ps.stem(list_of_words)
```
- **Lemmatization:** Spacy Tokenizer() does not do any semantic change in the document, only tokenize it. However, PorterStemmer() or .lemma_ method lemmatize the tokens.
```
doc = nlp(text)
for token in doc: token.lemma_.lower()
```

Notebook nlp_TextData-411a.ipynb uses similar techniques for yelp_coffeeshop_review_data.csv. To create a contrast we compare reviews with rating ==5 and those with rating==1 only.

### Libraries:
```
import requests, zipfile
from urllib.request import urlopen
from io import BytesIO
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import Counter
import squarify
import spacy
from spacy.tokenizer import Tokenizer
from nltk.stem import PorterStemmer
import numpy as np
```

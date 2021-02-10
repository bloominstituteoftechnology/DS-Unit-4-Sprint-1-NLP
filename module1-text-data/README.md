## NLP-TextData-u4s1m1:
Clone the repo to your local: `git clone https://github.com/skhabiri/ML-NLP.git`
For package management we are going to use conda.
Create the virtual environment: cd ML_NLP; `conda create -n ML_NLP python==3.7`
Now the environment is created in ~/opt/anaconda3/envs/ML_NLP. We can see the list of conda environments with `conda env list`. To remove an environment `conda env remove --name <conda-env>`. 
For installing the required packages, first we need to activate the environment `conda activate ML_NLP`. 
Install the packages listed in requirements.txt `pip install -r requirements.txt`. To list the installed packages in the environment: `conda list`. 
√ ML_NLP % cat requirements.txt
gensim==3.8.1
pyLDAvis==2.1.2
spacy==2.2.3
scikit-learn==0.22.2
seaborn==0.9.0
squarify==0.4.3
ipykernel
nltk
pandas
scipy
beautifulsoup4

ipykernel is Ipython Kernel, a python execution backend for Jupyter. In order to open the python environment from jupyter we add an ipython kernel referencing to conda environment by `python -m ipykernel install --user --name ML_NLP --display-name "ML_NLP (Python3.7)"`.
Next we need to download and install the models for spacy: `python -m spacy download en_core_web_md`, `python -m spacy download en_core_web_lg`
Now you can deactivate the environment and launch jupyter lab and select ML_NLP as the Ipython kernel.

### Notebooks
In nlp_TextData-411.ipynb we useDatafiniti_Amazon_Consumer_Reviews_of_Amazon_Products.zip dataset to tokenize and visualize ‘reviews.text’ column. 
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

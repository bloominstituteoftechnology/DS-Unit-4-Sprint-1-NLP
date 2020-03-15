Clean up dates in the review

Extra Words in the star_rating column

Analyze the corpus of text using text visualizations of token frequency. Try cleaning the data as much as possible. Try the following techniques: 
- Lemmatization
- Custom stopword removal

1. Keep in mind the attributes of good tokens.

The attributes of good tokens:

    - Should be stored in an iterable datastructure
       
    - Should be all the same case
     
    - Should be free of non-alphanumeric characters (ie punctuation,    whitespace)



2. Once you have a solid baseline, layer in the star rating in your visualization(s). 

3. Key part of this assignment - produce a write-up of the attributes of the best and worst coffee shops. 

4. Based on your analysis, what makes the best the best and the worst the worst. Use graphs and numbers from your analysis to support your conclusions. 

**There should be plenty of markdown cells!**

## How do we want to analyze these coffee shop tokens? 

- Overall Word / Token Count
- View Counts by Rating
- *Hint:* a 'bad' coffee shops has a rating betweeen 1 & 3 based on the distribution of ratings. A 'good' coffee shop is a 4 or 5.

## Can visualize the words with the greatest difference in counts between 'good' & 'bad'?

Couple Notes: 
- Rel. freq. instead of absolute counts b/c of different numbers of reviews
- Only look at the top 5-10 words with the greatest differences


## Stretch Goals

* Analyze another corpus of documents - such as Indeed.com job listings ;).
* Play with the Spacy API to
 - Extract Named Entities
 - Extracting 'noun chunks'
 - Attempt Document Classification with just Spacy
 - *Note:* This [course](https://course.spacy.io/) will be of interesting in helping you with these stretch goals. 
* Try to build a plotly dash app with your text data 
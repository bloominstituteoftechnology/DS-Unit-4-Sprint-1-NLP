Your primary assignment this afternoon is to achieve a minimum of 70% accuracy on the Kaggle competition. Once you have achieved 70% accuracy, please work on the following: 

1. Research "Sentiment Analysis". Provide answers in markdown to the following questions: 
    - What is "Sentiment Analysis"? 
        Determining the opinion or feeling of a piece of text.

    - Is Document Classification different than "Sentiment Analysis"? Provide evidence for your response
        It is and it isnt... Sentiment Analysis is a way of classifying documents. Sentiment analysis deals with
        determining the opinion or feeling of a piece of text. Using Sentiment analysis you could classify your documents.
        But, there are other ways that you can classify documents as well. You could classify documents based on a certain topic,
        or interest. Sentiment anlysis is just a way to classify documents.

    - How do create labeled sentiment data? Are those labels really sentiment?
        you would look at the text you are wanting to analyze and make a determination if you feel that text is positive, negative, 
        or neutral. You could add more labels as well, but that is a baseline. It is sentiment, but it might not be complete or capture the true feelings of the text. For example, if a text was annoyed, that could be looked at as negative. But, negative might not capture the entire sentiment of the text. Also a text could have a negative sentiment towards one thing but positive sentiment towards another. 

    - What are common applications of sentiment analysis?
        get the perception of consumers on a given brand. Look at reviews to see how customers feel about a product, or a service. market research to see how industryies are doing, or competitors. 

2. Research our why word embeddings worked better for the lecture notebook than on the whiskey competition.
    - This [text classification documentation](https://developers.google.com/machine-learning/guides/text-classification/step-2-5) from Google might be of interest

        the data set used in the assignment had a words/sample average of around 435. Given that it was a smaller number of words in each sample, we should use the n grams and the tf-idf to extract features to pass into our classifier model. 

        the data set used in the lecture had average of over 2000. Because it was a high number we would could use embeddings for a better result.
        
    - Neural Networks are becoming more popular for document classification. Why is that the case?
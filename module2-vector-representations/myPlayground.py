# This is my file to just play around in 
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


text = b"Job Requirements:\nConceptual understanding in Machine Learning models like Nai\xc2\xa8ve Bayes, K-Means, SVM, Apriori, Linear/ Logistic Regression, Neural, Random Forests, Decision Trees, K-NN along with hands-on experience in at least 2 of them\nIntermediate to expert level coding skills in Python/R. (Ability to write functions, clean and efficient data manipulation are mandatory for this role)\nExposure to packages like NumPy, SciPy, Pandas, Matplotlib etc in Python or GGPlot2, dplyr, tidyR in R\nAbility to communicate Model findings to both Technical and Non-Technical stake holders\nHands on experience in SQL/Hive or similar programming language\nMust show past work via GitHub, Kaggle or any other published article\nMaster's degree in Statistics/Mathematics/Computer Science or any other quant specific field.\nApply Now"

# This is my method that will pull out the 
# carriage return and the new lines from the string
def remove_carriage_return(text):
    text = str(text)
    
    newText = text.replace("\n", "")
    return newText






if __name__ == "__main__":
    
    print("---------")
    print("Trying to remove the carriage returns")
    text  = remove_carriage_return(text)
    print(text)
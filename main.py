import pandas as pd

def un_pig(string):
    
    unpigged = []
    word_list = string.split()

    for word in word_list:
        word = word[:-2]
        unpigged += [word[-1] + word[:-1]]
        
    return unpigged

df_gifts = pd.read_csv('iftsgay_2018_csv.csv')

df_gifts['Things'].apply(un_pig)
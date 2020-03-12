import pandas as pd
pd.set_option('display.max_columns', 50)

# Work with this data
gift_df = pd.read_csv('iftsgay_2018.csv')
recipients_df = pd.read_csv('aughtynay_roay_icenay_2018.csv')

# 1. Pig Latin to English translator
def translate_pl_english(words):
    try:
        if words.str.endswith('ay'):
            try:
                word = words.str.split('')
                word = word[-3] + word[:-3]
            except:
                word = word[-3] + word[:-3]
    except:
        print('Nope')
    pass

df1 = gift_df.apply(translate_pl_english)
df2 = recipients_df.apply(translate_pl_english)
breakpoint()

# 2. Anti-sad-ify the indicated "gift"
# your code here

# STRETCH: Assign gifts

# TO PASS TESTS:
# Run your code before hitting submit
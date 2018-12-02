from urllib.request import urlretrieve
import os

def parse_gz(path):
    import gzip
    g = gzip.open(path, 'rb')
    for l in g:
        yield eval(l)

def convert_to_df(path):
    import pandas as pd
    i = 0
    df = {}
    for d in parse_gz(path):
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient='index')


amazon_data_base_url = "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/"
local_base_path = "dataset/"

small_datasets = {
    "books": "reviews_Books_5.json.gz",
    "electronics": "reviews_Electronics_5.json.gz",
    "movies_and_tv": "reviews_Movies_and_TV_5.json.gz",
    "sports_and_outdoors": "reviews_Sports_and_Outdoors_5.json.gz"
}

def load_amazon_review_data(is_complete_data="False", category="sports_and_outdoors"):
    if is_complete_data == "False":
        data_name = small_datasets[category]
    filename = os.path.join(local_base_path, data_name)
    
    if not os.path.exists(filename):
        urlretrieve(amazon_data_base_url+small_datasets[category], 
                              filename)
    
    return convert_to_df(filename)

def tokenize(text):
    import string
    from nltk import word_tokenize
    tokenized = word_tokenize(text)
    no_punc = []
    for review in tokenized:
        line = "".join(char for char in review if char not in string.punctuation)
        no_punc.append(line)
    tokens = lemmatize(no_punc)
    return tokens

def lemmatize(tokens):
    from nltk.stem.wordnet import WordNetLemmatizer
    lmtzr = WordNetLemmatizer()
    lemma = [lmtzr.lemmatize(t) for t in tokens]
    return lemma
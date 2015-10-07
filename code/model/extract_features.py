import nltk.data
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import ast, gzip
import scipy.sparse

sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')

def num_sentences(text):
    return len(sent_detector.tokenize(text.strip()))

def num_tokens(text):
    return len(text.split())

def average_length_of_sentences(text):
    n_sent = num_sentences(text)
    n_tokens = num_tokens(text)
    return (1.0 * n_tokens) / (1.0 * n_sent)

def tf_idf_matrix(reviews):
    vectorizer = TfidfVectorizer(min_df=1)
    return vectorizer.fit_transform(reviews)

def create_feature_results_matrix(datapath):
    reviews = []
    helpfulness = []
    with gzip.open(datapath, 'r') as f:
        for l in f:
            rev = ast.literal_eval(l.strip())
            if not rev['helpful']:
                continue
            reviews.append(rev['reviewText'])
            helpfulness.append((1.0 * rev['helpful'][0]) / (1.0 * rev['helpful'][1]))
    X_tfidf = tf_idf_matrix(reviews)
    df = pd.DataFrame(columns=('reviewText', 'num_sentences', 'num_tokens', 'avg_length', 'helpfulness'))
    for i in range(len(reviews)):
        reviewText = reviews[i]
        n_sent = num_sentences(reviewText)
        n_tokens = num_tokens(reviewText)
        avg_length = average_length_of_sentences(reviewText)
        df.loc[i] = [reviewText, num_sentences, num_tokens, avg_length, helpfulness[i]]

    y = df['helpfulness']
    df = df[['num_sentences', 'num_tokens', 'avg_length']]
    X_other = df.as_matrix()
    X = scipy.sparse.hstack([X_tfidf, X_other])
    return X, y







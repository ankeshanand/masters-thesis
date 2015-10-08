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
    if n_sent == 0:
        n_sent += 1
        print text
    n_tokens = num_tokens(text)
    return (1.0 * n_tokens) / (1.0 * n_sent)

def tf_idf_matrix(reviews):
    vectorizer = TfidfVectorizer(min_df=4, stop_words='english')
    return vectorizer.fit_transform(reviews)

def create_feature_results_matrix(datapath):
    reviews = []
    helpfulness = []
    n_sentences, n_tokens, avg_length = [], [], []
    with gzip.open(datapath, 'r') as f:
        for l in f:
            rev = ast.literal_eval(l.strip())
            if not rev['helpful']:
                continue
            if rev['helpful'][1] < 5:
                continue
            if rev['reviewText'] == '':
                continue
            reviews.append(rev['reviewText'])
            helpfulness.append((1.0 * rev['helpful'][0]) / (1.0 * rev['helpful'][1]))
            text = rev['reviewText']
            n_sentences.append(num_sentences(text))
            n_tokens.append(num_tokens(text))
            avg_length.append(average_length_of_sentences(text))

    print 'Parsing complete'
    print len(reviews)
    X_tfidf = tf_idf_matrix(reviews)
    print 'Computed the tf-idf matrix.'
    df = pd.DataFrame({'reviewText': reviews, 'n_sentences': n_sentences, 'n_tokens': n_tokens, 'avg_length': avg_length, 'helpfulness': helpfulness})

    print 'Created the other features dataframe.'
    y = df['helpfulness']
    df = df[['n_sentences', 'n_tokens', 'avg_length']]
    X_other = df.as_matrix()
    X = scipy.sparse.hstack([X_tfidf, X_other])
    print 'Stacked features, returning from method.'
    return X, y







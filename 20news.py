import numpy as np
import pandas as pd

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer

categories = [
    'sci.space',
    'sci.med',
    'comp.graphics',
    'rec.sport.baseball'
]

data = fetch_20newsgroups(
    subset='train',
    categories=categories,
    remove=('headers', 'footers', 'quotes')
)

print("Number of documents:", len(data.data))

vectorizer = CountVectorizer(
    stop_words='english',
    min_df=2
)

X = vectorizer.fit_transform(data.data)

print("Vocabulary size:", len(vectorizer.vocabulary_))


word_count = np.array(X.sum(axis=0)).flatten()
vocabulary = vectorizer.get_feature_names_out()

N = word_count.sum()
K = len(word_count)

print("Total word count:", N)
print("Number of vocabulary words:", K)

mle = word_count / N

def map_estimation(counts, alpha):
    N = counts.sum()
    K = len(counts)

    numerator = counts + alpha - 1
    denominator = N + K * (alpha - 1)

    return numerator / denominator



map_alpha_1 = map_estimation(word_count, 1.0)
map_alpha_1_1 = map_estimation(word_count, 1.1)
map_alpha_2 = map_estimation(word_count, 2.0)



results = pd.DataFrame({
    'Word': vocabulary,
    'Count': word_count,
    'MLE': mle,
    'MAP_alpha_1': map_alpha_1,
    'MAP_alpha_1.1': map_alpha_1_1,
    'MAP_alpha_2': map_alpha_2
})

results = results.sort_values(
    by='Count',
    ascending=False
)

print(results.head(20))

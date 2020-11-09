from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from scipy.stats import spearmanr
import nltk.data
import re
import tensorflow as tf
import logging
from gensim.models import word2vec
from pythainlp import word_vector
from sklearn.metrics import cohen_kappa_score
import timeit
from sklearn.model_selection import train_test_split
import pythainlp
from pythainlp import sent_tokenize, word_tokenize
from pythainlp.corpus import thai_stopwords

xl_workbook = pd.ExcelFile('quiz.xlsx')
df_all = xl_workbook.parse("training_set")
df_all = df_all.drop('Timestamp', 1)

X_train, X_test, y_train, y_test = train_test_split(df_all['essay'], df_all['Score'], test_size=0.10)
y_train = y_train.values.reshape(len(y_train), 1)
y_test = y_test.values.reshape(len(y_test), 1)

def essay_to_wordlist(essay_v, remove_stopwords):
    essay_v = re.sub("[^ก-ฮฤ-์]", " ", essay_v)
    words = word_tokenize(essay_v, engine="newmm", keep_whitespace=False)
    if remove_stopwords:
        stops = set(thai_stopwords())
        words = [w for w in words if not w in stops]
    return words

def essay_to_sentences(essay_v, remove_stopwords):
    raw_sentences = sent_tokenize(essay_v.strip())
    sentences = []
    for raw_sentence in raw_sentences:
        if len(raw_sentence) > 0:
            sentences.append(essay_to_wordlist(raw_sentence, remove_stopwords))
    return sentences

sentences = []

for essay_v in X_train:
    sentences += essay_to_sentences(essay_v, remove_stopwords=True)
for essay_v in X_test:
    sentences += essay_to_sentences(essay_v, remove_stopwords=True)
    
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

num_features = 300
min_word_count = 40
num_workers = 4
context = 10
downsampling = 1e-3

print("Training model...")
model = word2vec.Word2Vec(sentences, workers=num_workers, size=num_features, min_count=min_word_count, window=context,
                          sample=downsampling)

model.init_sims(replace=True)

model_name = "300features_40minwords_10context"
model.save(model_name)

def makeFeatureVec(words, model, num_features):
    featureVec = np.zeros((num_features,), dtype="float32")
    nwords = 0
    index2word_set = set(model.wv.index2word)
    for word in words:
        if word in index2word_set:
            nwords = nwords + 1
            featureVec = np.add(featureVec, model[word])
    featureVec = np.divide(featureVec, nwords)
    if (nwords==0):
        print(words)
    return featureVec


def getAvgFeatureVecs(essays, model, num_features):
    counter = 0
    essayFeatureVecs = np.zeros((len(essays), num_features), dtype="float32")
    for essay in essays:
        essayFeatureVecs[counter] = makeFeatureVec(essay, model, num_features)
        counter = counter + 1
    return essayFeatureVecs

print("Creating average feature vecs for Training Essays")
clean_train_essays = []
for essay_v in X_train:
    clean_train_essays.append(essay_to_wordlist(essay_v, remove_stopwords=True))
trainDataVecs = getAvgFeatureVecs(clean_train_essays, model, num_features)

clean_test_essays = []
for essay_v in X_test:
    clean_test_essays.append(essay_to_wordlist(essay_v, remove_stopwords=True))
testDataVecs = getAvgFeatureVecs(clean_test_essays, model, num_features)
print("done!")


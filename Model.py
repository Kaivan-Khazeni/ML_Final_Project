import pandas as pd
import numpy as np
import pickle

# NLTK Libraries and Functions
import nltk
import nltk
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

import Data_Processing as data

#Import TextBlob
from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer
from textblob.classifiers import NaiveBayesClassifier

global_train_data =  pd.DataFrame()

def create_and_save_model(train_data,test_data):
    model = NB_Model(train_data, test_data)
    saved_model = "model.pkl"
    with open(saved_model, 'wb') as file:
        pickle.dump(model, file)

    #returning file name
    return saved_model

def load_model(file_name):
    with open(file_name, 'rb') as file:
        model = pickle.load(file)
    return model

def split_data(df, random=False):
    if random == False:
        train = df.sample(frac=0.7)
        test = df.drop(train.index)

        mono_train = data.get_monologue_df(train)
        mono_test = data.get_monologue_df(test)

        sentence_train = data.get_sentence_df(train)
        sentence_test = data.get_sentence_df(test)

        return mono_train,mono_test,sentence_train,sentence_test
    else:
        mono_df = data.get_monologue_df(df)
        sentence_df = data.get_sentence_df(df)

        mono_train = mono_df.sample(frac=0.7)
        mono_test = mono_df.drop(mono_train.index)

        sentence_train = sentence_df.sample(frac=0.7)
        sentence_test = sentence_df.drop(sentence_train.index)

        return mono_train,mono_test,sentence_train,sentence_test


def run_experiment_same(df):
    mono_train, mono_test, sentence_train, sentence_test = split_data(df, random=False)
    mono_model = NB_Model(mono_train,mono_test)
    mono_accuracy = mono_model.accuracy()

    sentence_model = NB_Model(sentence_train, sentence_test)
    sentence_accuracy = sentence_model.accuracy()


    return mono_accuracy,sentence_accuracy


def run_experiment_random(df):
    mono_train, mono_test, sentence_train, sentence_test = split_data(df, random=True)
    mono_model = NB_Model(mono_train, mono_test)
    mono_accuracy = mono_model.accuracy()

    sentence_model = NB_Model(sentence_train, sentence_test)
    sentence_accuracy = sentence_model.accuracy()

    return mono_accuracy, sentence_accuracy

def multi_classification(label):
    if len(label) != 3:
        if np.array_equal(label, np.array(['neu', 'pos'])) or np.array_equal(label, np.array(['pos', 'neu'])):
            return 'semi-positive'
        if np.array_equal(label, np.array(['neu', 'neg'])) or np.array_equal(label, np.array(['neg', 'neu'])):
            return 'semi-negative'
    else:
        return label
def classify_speaker_sentiment(df,file_name):
    model = load_model(file_name)
    #Model after opening the picke file
    test_predictions = model.classify()
    df['sentiment'] = test_predictions

    speaker_sent = df.groupby(['Call ID', 'Speaker Number'])['sentiment'].agg(pd.Series.mode).reset_index()
    speaker_sent['sentiment'] = speaker_sent['sentiment'].apply(multi_classification)
    return speaker_sent


class NB_Model:
    def __init__(self, train_data, test_data):
        self.cl = NaiveBayesClassifier(list(tuple(val) for val in tuple(train_data.values)))
        self.test_data = test_data

    def accuracy(self):
        self.accuracy = self.cl.accuracy(list(tuple(val) for val in tuple(self.test_data.values)))
        return self.accuracy

    def classify(self):
        self.test_data['prediction'] = self.test_data.sent.apply(self.cl.classify)
        return self.test_data['prediction']















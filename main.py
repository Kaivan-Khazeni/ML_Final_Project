import Data_Processing as data
import Topic_Analysis as topic
# Libraries for Dataframes
import pandas as pd
import numpy as np
# NLTK Libraries and Functions
import nltk
from nltk.corpus import sentiwordnet as swn, stopwords, wordnet
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

import Model
# Misc.
import re
from pprint import pprint
from collections import Counter
# Import TextBlob
from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer
import Model as model

if __name__ == '__main__':
    df = data.get_call_df("Data/Gona_Call_Transcripts.xlsx")
    call_id = list(df['Call ID'].unique())
    call_id = call_id[0:250]

    # HERE ARE THE STATEMENTS TO RUN THE EXPERIMENTS

    # 1. FINDING ACCURACY OF SAME VERSUS RANDOM SAMPLING DATA
    """
    call_id = list(df['Call ID'].unique())
    call_id = call_id[0:250]

    Splitting dataset for monologue or sentence train and testing

    mono_train, mono_test, sentence_train, sentence_test = model.split_data(df, random=False)
    mono_same_accuracy,sentence_same_accuracy,saved_model = model.run_experiment_same(df)
    mono_random_accuracy,sentence_random_accuracy = model.run_experiment_random(df)
    print("Results of experiment 1:  Monologue Accuracy - ", mono_same_accuracy, "\n Sentence Accuracy -", sentence_same_accuracy)
    print("Results of experiment 2:  Monologue Accuracy - ", mono_random_accuracy, "\n Sentence Accuracy -", sentence_random_accuracy)
    """

    # 2.
    """
    call_id_train = call_id[101:250]
    call_id_test = call_id[0:100]

    train_df = df[df['Call ID'].isin(call_id_train)]
    test_df = df[df['Call ID'].isin(call_id_test)]

    sentence_train_df, call_speaker_train = data.get_sentence_df(train_df, train=True)
    sentence_test_df, call_speaker_test = data.get_sentence_df(test_df, train=False)
    file_name = model.create_and_save_model(sentence_train_df, sentence_test_df)
    speaker_sentiment = model.classify_speaker_sentiment(call_speaker_test, file_name)
    print(speaker_sentiment.iloc[0:20])
    """

    # 3.
    """
    topic_df = pd.read_excel('Data/gong.xlsx')
    print(topic.call_topic_sentiment(topic_df).head(5))
    """


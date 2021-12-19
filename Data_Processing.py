#Libraries for Dataframes
import pandas as pd
import numpy as np

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
# NLTK Download neccessary features
nltk.download(['sentiwordnet', 'wordnet'])
nltk.download(['stopwords', 'punkt', 'vader_lexicon', 'shakespeare', 'opinion_lexicon', 'averaged_perceptron_tagger', 'wordnet'])
from nltk.corpus import sentiwordnet as swn, stopwords, wordnet
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
""
#Misc.
import re
from pprint import pprint
from collections import Counter
#Import TextBlob
from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer


#Importing and cleaning the dataset
def get_call_df(path):
    # Already ordered by calls, and order in which conversation happened
    df = pd.read_excel(path)
    # Keep needed columns
    call_df = df[['Call ID', 'Speaker Number', 'Monologue Number', 'Transcript', '# of Phrases in Monologue']]
    # Sort by speaker, monologue
    call_df = call_df.sort_values(by=['Call ID', 'Speaker Number', 'Monologue Number'])
    return call_df


#Label each monologue using textblob's NaiveBayesAnalyzer
#Use a Classifying TextBlob. We will use a Naive Bayes Classifier.
#Refer to Documentation...
# "https://textblob.readthedocs.io/en/dev/classifiers.html" to get definitions of what TextBlob is, how it works.
#Each Monologue can be its own TextBlob.
def get_monologue_df(df):
    monologue_df = df[['Call ID', 'Monologue Number', 'Transcript']]
    monologue_df['label'] = ''
    monologue_df = label_data_A(monologue_df.copy())
    monologue_df['label'] = monologue_df['label'].apply(numeric_to_categorical)

    return monologue_df[['Transcript', 'label']]

def label_data_A(df):

    for i, text in df.Transcript.iteritems():
        label = TextBlob(text)
        df['label'][i]= label.sentiment.polarity
    return df
def label_data_B(df):

    for i, text in df.sent.iteritems():
        label = TextBlob(text)
        df['label'][i] = label.sentiment.polarity
    return df

# Create function that converts numerical labels -> categorical
def numeric_to_categorical(num):
  if(num < 0 and num >= -1):
    return 'neg'
  elif num == 0:
    return 'neu'
  elif(num > 0 and num <= 1):
    return 'pos'


def get_sentence_df(df,train):

    sent_df = df.copy()
    sent_df['Transcript'] = sent_df.Transcript.apply(sent_tokenize)
    sentences = list(sent_df['Transcript'])
    call_ids = list(sent_df['Call ID'])
    speaker_ids = list(sent_df['Speaker Number'])

    total_sentences,call_speaker_info = flatten_list(call_ids,speaker_ids,sentences)
    #DF holding the caller ID, speaker number, and sent. This will be used to find the mode
    #of the predictions per speaker for each call
    call_speaker_sent = pd.DataFrame(
        data={'Call ID': [c for c,_ in call_speaker_info], 'Speaker Number': [s for _,s in call_speaker_info], 'sent': total_sentences})

    tokenized_sentences = [word_tokenize(sent) for sent in total_sentences]
    num_of_words = [len(sent) for sent in tokenized_sentences]
    sentences_df = pd.DataFrame(
        data={'sent': total_sentences, 'tokenized_sent': tokenized_sentences, 'num_of_words': num_of_words})
    if train == True:
        sentences_df = sentences_df[sentences_df['num_of_words'] > 2]
    sentences_df['label'] = ''
    sentences_df = label_data_B(sentences_df.copy())
    sentences_df['label'] = sentences_df['label'].apply(numeric_to_categorical)

    return sentences_df[['sent','label']],call_speaker_sent

def flatten_list(call_ids,speaker_ids, _2d_list):
    flat_list = []
    call_speaker_ids = []
    # Iterate through the outer list
    for i,element in enumerate(_2d_list):
        if type(element) is list:
            # If the element is of type list, iterate through the sublist
            for item in element:
                flat_list.append(item)
                call_speaker_ids.append((call_ids[i],speaker_ids[i]))
        else:
            flat_list.append(element)
            call_speaker_ids.append((call_ids[i], speaker_ids[i]))

    return flat_list,call_speaker_ids



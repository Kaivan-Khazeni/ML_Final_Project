from textblob import TextBlob
import pandas as pd
import pandas as pd
import numpy as np
import nltk
from pprint import pprint
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
# grabbing a part of speech function:
import nltk, re
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from collections import Counter


def get_part_of_speech(word):
    probable_part_of_speech = wordnet.synsets(word)
    pos_counts = Counter()
    pos_counts["n"] = len([item for item in probable_part_of_speech if item.pos() == "n"])
    pos_counts["v"] = len([item for item in probable_part_of_speech if item.pos() == "v"])
    pos_counts["a"] = len([item for item in probable_part_of_speech if item.pos() == "a"])
    pos_counts["r"] = len([item for item in probable_part_of_speech if item.pos() == "r"])
    most_likely_part_of_speech = pos_counts.most_common(1)[0][0]
    return most_likely_part_of_speech


def preprocess_text(text):
    cleaned = re.sub('\W+', ' ', text).lower()
    tokenized = word_tokenize(cleaned)
    normalizer = WordNetLemmatizer()
    normalized = [normalizer.lemmatize(token, get_part_of_speech(token)) for token in tokenized]
    return normalized


def text_to_bow(tokens):
    bow_dictionary = {}
    # tokens = preprocess_text(some_text)
    for token in tokens:
        if token in bow_dictionary:
            bow_dictionary[token] += 1
        else:
            bow_dictionary[token] = 1
    return bow_dictionary


def get_noun_frequencies(callID, df):
    call_df = df[df['Call ID'] == callID]
    call_df = call_df.reset_index(drop=True)
    stop_words = stopwords.words('english')
    full_transcript_text = ""
    for text in call_df['Transcript']:
        full_transcript_text = full_transcript_text + " " + text

    tokens = preprocess_text(full_transcript_text)
    filtered = [word for word in tokens if word not in stop_words]
    bow_dictionary = text_to_bow(filtered)

    nouns = {}
    for key, val in bow_dictionary.items():
        pos = get_part_of_speech(key)
        if pos == 'n':
            nouns[key] = val

    return sorted(nouns.items(), key=lambda x: x[1], reverse=True)


def get_topics(df):
    topic_dict = {}
    for id in df['Call ID'].unique():
        temp_topic = get_noun_frequencies(id, df)[0][0]
        test = get_noun_frequencies(id, df)
        words_of_interest = ['policy', 'payement', 'credit', 'insurance', 'account', 'cancellation', 'deductable',
                             'home', 'life', 'auto']
        # Here we only use the key words that appear from the list, that way we define the conversation topic
        # via common words with insurance as shown in the array above
        for i in range(len(test)):
            if test[i][0] in words_of_interest:
                temp_topic = test[i][0]
                break
        topic_dict[id] = temp_topic
    return topic_dict


def call_topic_sentiment(topic_df):
    topic_dict = get_topics(topic_df)
    topic_sentiment = []
    for key, value in topic_dict.items():
        # Slice Dataframe for each Call, Topic
        temp_df = topic_df[topic_df['Call ID'] == key]
        temp_df = temp_df[temp_df['Transcript'].str.contains(value)]

        topic_sentiment.append(find_sentiment(temp_df))

    ret_df = pd.DataFrame.from_dict(topic_dict, orient='index', columns=['Topic']).reset_index(0)
    ret_df['Sentiment'] = topic_sentiment
    ret_df['Sentiment'] = ret_df['Sentiment'].apply(numeric_to_categorical)
    ret_df = ret_df.rename(columns={'index': 'Call ID', 'Topic': 'Topic', 'Sentiment': 'Sentiment'})

    return ret_df


def find_sentiment(df):
    df['sentiment'] = ''
    for i in range(len(df['Transcript'])):
        label = TextBlob(df['Transcript'].values[i])
        df['sentiment'].iloc[i] = label.sentiment.polarity

    return np.mean(df['sentiment'])

# Create function that converts numerical labels -> categorical
def numeric_to_categorical(num):
    if (num < 0 and num >= -1):
        return 'neg'
    elif num == 0:
        return 'neu'
    elif (num > 0 and num <= 1):
        return 'pos'


import os.path

import pandas as pd
import numpy as np
import re
import nltk
from nltk import word_tokenize
from nltk.stem import PorterStemmer
import spacy
import string
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, recall_score, precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, Flatten
from utils import *

PUNCT_TO_REMOVE = string.punctuation
STOPWORDS = set(stopwords.words('english'))
STEMMER = PorterStemmer()
SLANG_WORDS = get_slang_words()
emoj = re.compile("["
                  u"\U0001F600-\U0001F64F"  # emoticons
                  u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                  u"\U0001F680-\U0001F6FF"  # transport & map symbols
                  u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                  u"\U00002500-\U00002BEF"  # chinese char
                  u"\U00002702-\U000027B0"
                  u"\U00002702-\U000027B0"
                  u"\U000024C2-\U0001F251"
                  u"\U0001f926-\U0001f937"
                  u"\U00010000-\U0010ffff"
                  u"\u2640-\u2642"
                  u"\u2600-\u2B55"
                  u"\u200d"
                  u"\u23cf"
                  u"\u23e9"
                  u"\u231a"
                  u"\ufe0f"  # dingbats
                  u"\u3030"
                  "]+", re.UNICODE)


def remove_punctuation(text):
    return text.translate(str.maketrans('', '', PUNCT_TO_REMOVE))


def remove_stopwords(text):
    return " ".join([word for word in str(text).split() if word not in STOPWORDS])


def lemmatizing(text):

    lemmatized = [token.lemma_ for token in text]
    return ' '.join([STEMMER.stem(token) for token in lemmatized])


def isEnglish(s):
    try:
        s.encode(encoding='utf-8').decode('ascii')
    except UnicodeDecodeError:
        return 0
    else:
        return 1


def remove_emojis(data):
    return re.sub(emoj, '', data)


def remove_slang_words(text):

    return " ".join([word for word in str(text).split() if word not in SLANG_WORDS])


def get_top_n_words(corpus, n=None):
    vec = CountVectorizer().fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    return words_freq[:n]


def data_preprocessing(df: pd.DataFrame) -> pd.DataFrame:

    df.drop_duplicates(subset=['message'], inplace=True)
    df["message"] = df["message"].astype(str)
    df["message"] = df["message"].str.lower()
    df["message"] = df["message"].apply(lambda text: remove_punctuation(text))
    df["message"] = df["message"].apply(lambda text: remove_stopwords(text))
    df['message'] = df["message"].apply(lambda text: remove_slang_words(text))
    df['message'] = df["message"].apply(lambda text: remove_emojis(text))
    # df["message"] = df["message"].apply(lambda text: lemmatizing(nlp(text)))
    df['msg_len'] = df["message"].apply(lambda text: len(word_tokenize(text)))

    df.drop(df[df.message == ''].index, inplace=True)
    df.drop_duplicates(subset=['message'], inplace=True)
    df = df.sample(frac=1)

    # df['is_english'] = df["message"].apply(lambda text: isEnglish(text))
    num_banned_msgs = df[df.banned == 1].shape[0]
    drop_indices = np.random.choice(df[df.banned == 0].index, df[df.banned == 0].shape[0] - num_banned_msgs, replace=False)
    df.drop(drop_indices, inplace=True)
    # clean_df = df_subset.drop(df_subset[df_subset.is_english == 0].index)

    # test_df = df[['message_wo_punct', 'banned']]
    # test_df.reset_index(inplace=True)
    # test_df.drop(columns=['index'], axis=1, inplace=True)
    # # t_df = df[['message_wo_punct', 'banned']]
    # # t_df = t_df.loc[t_df.banned == 1]
    # # test_df = test_df.append(t_df)
    # test_df = test_df.sample(frac=1)
    # # test_df["lemmatized_msgs"] = test_df["message_wo_punct"].apply(lambda text: lemmatizing(nlp(text)))
    # test_df['clean_msgs'] = test_df.message_wo_punct.apply(lambda text: remove_emojis(text))
    # test_df['is_english'] = test_df.clean_msgs.apply(lambda text: isEnglish(text))
    # clean_df = test_df.drop(test_df[test_df.is_english == 0].index)
    df.to_csv('./processed_data.csv', sep=',', index=False)
    print('// Data Statistics------------')
    print('Total rows: ', df.shape[0])
    print('Number of banned messages: ', df[df.banned == 1].shape[0])
    return df


def printing_results(y_test, *models):
    for model_name, predictions in models:
        print(f'\n----Model: {model_name}')
        # print("Report: ", classification_report(y_true=y_test, y_pred=prediction))
        # print("\nMatrix: ", confusion_matrix(y_true=y_test, y_pred=prediction))
        print("Accuracy: ", accuracy_score(y_true=y_test, y_pred=predictions))
        print("F1-Score: ", f1_score(y_true=y_test, y_pred=predictions))
        print("Recall: ", recall_score(y_true=y_test, y_pred=predictions))
        print("Precision: ", precision_score(y_true=y_test, y_pred=predictions))


def NN_with_embeddings(X_train, X_test, y_train, vocab_size, max_length):

    encoded_docs_train = [one_hot(d, vocab_size) for d in X_train]
    encoded_docs_test = [one_hot(d, vocab_size) for d in X_test]
    padded_docs_train = pad_sequences(encoded_docs_train, maxlen=max_length, padding='post')
    padded_docs_test = pad_sequences(encoded_docs_test, maxlen=max_length, padding='post')

    model = Sequential()
    model.add(Embedding(vocab_size, 64, input_length=max_length))
    model.add(Flatten())
    model.add(Dense(200, activation='relu'))
    model.add(Dropout(0.9))
    model.add(Dense(20, activation='relu'))
    model.add(Dropout(0.6))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(padded_docs_train, y_train, validation_split=0.20, epochs=12, verbose=0)

    prediction = np.array([round(x[0], 0) for x in model.predict(padded_docs_test)]).transpose()

    return 'Embeddings Neural Network', prediction


def implement_RF(X_train, X_test, y_train):
    classifier = RandomForestClassifier()
    classifier.fit(X_train, y_train)

    return 'Random Forest', classifier.predict(X_test)


def implement_LR(X_train, X_test, y_train):
    classifier = LogisticRegression()
    classifier.fit(X_train, y_train)

    return 'Logistic Regression', classifier.predict(X_test)


def implement_NN(X_train, X_test, y_train):
    model_dim = X_train.shape[1]

    model = Sequential()
    model.add(Dense(1200, input_dim=model_dim, activation='relu'))
    model.add(Dropout(0.9))
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(0.5))
    # model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # model.summary()

    model.fit(X_train, y_train, validation_split=0.20, epochs=14, verbose=1)
    prediction = np.array([round(x[0], 0) for x in model.predict(X_test)]).transpose()

    return 'FF Neural Network', prediction


def main():

    if os.path.isfile('./processed_data.csv'):
        test_df = pd.read_csv('./processed_data.csv', sep=',')
    else:
        df = pd.read_csv('./pgl_chat_data.csv')
        test_df = data_preprocessing(df)
    # df = pd.read_csv('./pgl_chat_data.csv')
    # test_df = data_preprocessing(df)

    messages = test_df.message.values
    y = test_df['banned'].values
    msgs_train, msgs_test, y_train, y_test = train_test_split(messages, y, test_size=0.25, random_state=100)
    max_length = test_df['msg_len'].max()

    vectorizer = CountVectorizer(analyzer="word")
    vectorizer.fit(messages)
    vocab_size = len(vectorizer.vocabulary_)

    X_train = vectorizer.transform(msgs_train).toarray().astype('float32')
    X_test = vectorizer.transform(msgs_test).toarray().astype('float32')

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.fit_transform(X_test)

    # ENN, ENN_preds = NN_with_embeddings(msgs_train, msgs_test, y_train, vocab_size, max_length)
    # LR, LR_preds = implement_LR(X_train_scaled, X_test_scaled, y_train)
    NN, NN_preds = implement_NN(X_train_scaled, X_test_scaled, y_train)
    RF, RF_preds = implement_RF(X_train_scaled, X_test_scaled, y_train)

    printing_results(y_test, (NN, NN_preds), (RF, RF_preds))


if __name__ == '__main__':
    main()

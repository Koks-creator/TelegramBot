import json
from random import choice
import os
import pickle
import string
from collections import Counter
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
import pandas as pd
from nltk.stem import WordNetLemmatizer
import numpy as np


class RemovePunctuation(BaseEstimator, TransformerMixin):
    def __init__(self, column_name: str):
        self.column_name = column_name
        self.translator = str.maketrans("", "", string.punctuation)

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X[self.column_name] = X[self.column_name].apply(lambda x: x.translate(self.translator))
        return X


class LemmatizeSents(BaseEstimator, TransformerMixin):
    def __init__(self, column_name: str):
        self.column_name = column_name
        self.lemmatizer = WordNetLemmatizer()

    def lemmatize_data(self, sent: str) -> str:
        filtered_text = [self.lemmatizer.lemmatize(word.lower()) for word in sent.split()]
        return " ".join(filtered_text)

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X[self.column_name] = X[self.column_name].apply(lambda x: self.lemmatize_data(x))
        return X


class OneHotEncodingLabels(BaseEstimator, TransformerMixin):
    def __init__(self, labels_column: str, label_encoder_path: str):
        self.labels_column = labels_column
        self.label_encoder_path = label_encoder_path
        self.le = preprocessing.LabelEncoder()

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        classes_list = X[self.labels_column].to_list()
        y = self.le.fit_transform(classes_list)

        if not os.path.isfile(self.label_encoder_path):
            with open(self.label_encoder_path, "wb") as f:
                pickle.dump(self.le, f)

        y = tf.keras.utils.to_categorical(y, num_classes=len(set(classes_list)))

        X[self.labels_column] = [y[i] for i in range(len(y))]

        return X


class TokenizeAndPad(BaseEstimator, TransformerMixin):
    def __init__(self, column_name: str, max_length: int, tokenizer_path: str, unique_words_path: str):
        self.column_name = column_name
        self.max_length = max_length
        self.tokenizer_path = tokenizer_path
        self.unique_words_path = unique_words_path

    @staticmethod
    def count_words(words_list: list) -> Counter:
        count = Counter()

        for word in words_list:
            for word in word.split():
                count[word] += 1

        return count

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        x_data = X[self.column_name].to_numpy()

        if not os.path.isfile(self.tokenizer_path):
            unique_words = self.count_words(X[self.column_name].to_list())
            tokenizer = Tokenizer(num_words=len(unique_words))
            tokenizer.fit_on_texts(x_data)

            with open(self.tokenizer_path, "wb") as f:
                pickle.dump(tokenizer, f)

            with open(self.unique_words_path, "wb") as f:
                pickle.dump(unique_words, f)
        else:
            with open(self.tokenizer_path, "rb") as f:
                tokenizer = pickle.load(f)

        x_data_tokenized = tokenizer.texts_to_sequences(x_data)
        x_data_padded = pad_sequences(x_data_tokenized, maxlen=self.max_length, padding="post", truncating="post")

        X[self.column_name] = [x_data_padded[i] for i in range(len(x_data_padded))]
        return X


class SaveDataset(BaseEstimator, TransformerMixin):
    def __init__(self, output_path: str):
        self.output_path = output_path

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> dict:
        X_data = np.array(X["X"].to_list())
        y_data = np.array(X["y"].to_list())

        data = {
            "X": X_data,
            "y": y_data
        }
        if not os.path.isfile(rf"{self.output_path}"):
            with open(self.output_path, "wb") as f:
                pickle.dump(data, f)

        return data


class MakePredictions(BaseEstimator, TransformerMixin):
    def __init__(self, model_path: str, target_column_name: str, label_encoder_path: str, intents_path: str):
        self.model_path = model_path
        self.target_column_name = target_column_name
        self.label_encoder_path = label_encoder_path
        self.intents_path = intents_path

    def decode_class(self, class_id: np.array) -> str:
        with open(self.label_encoder_path, "rb") as f:
            label_encoder = pickle.load(f)

        return label_encoder.inverse_transform([class_id])

    def get_answer(self, pred: np.array) -> str:
        with open(self.intents_path) as f:
            intents = json.load(f)["intents"]

        for intent in intents:
            tag = intent["intent"]
            if tag == pred:
                return choice(intent["responses"])

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        model = load_model(self.model_path)
        data = X[self.target_column_name].to_list()

        preds_raw = model.predict(np.array(data))

        predictions = [self.decode_class(np.argmax(pred))[0] for pred in preds_raw]
        answers = [self.get_answer(pred_class) for pred_class in predictions]

        X["Classes"] = predictions
        X["Answers"] = answers
        return X


def setup_preprocessing_pipeline(x_column_name: str, labels_column: str, max_length: int, label_encoder_path: str,
                                 tokenizer_path: str, unique_words_path: str, data_output_path: str) -> Pipeline:

    preprocessing_pipeline = Pipeline([
        ("Remove Punctuation", RemovePunctuation(column_name=x_column_name)),
        ("Lemmatize Sentiments", LemmatizeSents(column_name=x_column_name)),
        ("One Hot Encoding Labels", OneHotEncodingLabels(labels_column=labels_column, label_encoder_path=label_encoder_path)),
        ("Tokenize And Padding", TokenizeAndPad(column_name=x_column_name, max_length=max_length, tokenizer_path=tokenizer_path,
                                                unique_words_path=unique_words_path)),
        ("Save Dataset", SaveDataset(output_path=data_output_path)),
    ])

    return preprocessing_pipeline


def setup_prediction_pipeline(target_column_name: str, max_length: int, tokenizer_path: str, label_encoder_path: str,
                              unique_words_path: str, model_path: str, intents_path: str) -> Pipeline:

    prediction_pipeline = Pipeline([
        ("Remove Punctuation", RemovePunctuation(column_name=target_column_name)),
        ("Lemmatize Sentiments", LemmatizeSents(column_name=target_column_name)),
        ("Tokenize And Padding", TokenizeAndPad(column_name=target_column_name,
                                                max_length=max_length,
                                                tokenizer_path=tokenizer_path,
                                                unique_words_path=unique_words_path)),
        ("Make Predictions",  MakePredictions(target_column_name=target_column_name,
                                              model_path=model_path,
                                              label_encoder_path=label_encoder_path,
                                              intents_path=intents_path))

    ])

    return prediction_pipeline


if __name__ == '__main__':
    with open("../intents4.json") as f:
        intents = json.load(f)

    # Create data frame

    X = []
    y = []

    for intent in intents["intents"]:
        for pattern in intent["text"]:
            X.append(pattern)
            y.append(intent["intent"])

    df = pd.DataFrame(
        {
            "X": X,
            "y": y
        }
    )

    preprocessing_pipeline = setup_preprocessing_pipeline(
        column_name="X",
        labels_column="y",
        max_length=10,
        label_encoder_path="data/label_encoder.pkl",
        tokenizer_path="data/tokenizer.pkl",
        unique_words_path="data/words.pkl",
        data_output_path="data/train_data.pkl"
    )

    df = preprocessing_pipeline.fit_transform(df)


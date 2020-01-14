from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import ComplementNB
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, hamming_loss

import sys
import sqlite3
import pickle

import pandas as pd

# NLP import
import spacy
nlp = spacy.load("en_core_web_sm")


def load_data(database_filepath):
    """ Load data from 'message' table of the input SQLite database.

    Args:
        database_filepath (str): path to SQLite database file

    Returns:
        X (pd.DataFrame): features
        Y (pd.DataFrame): targets
        category_names ([str]): list of message category names
    """

    conn = sqlite3.connect(database_filepath)
    df = pd.read_sql_query('SELECT * FROM message', conn, index_col='id')
    X = df.message
    Y = df.loc[:, 'related':]
    category_names = Y.columns.values
    return X, Y, category_names


def tokenize(text):
    """ Return lowercase lemmatized tokens of the input text.
        Remove punctuation, spacy.stop_words, and pronouns.

    Args:
        text (str): input text
    Returns:
        tokens ([str]): list of lower case lemmatized tokens
    """

    doc = nlp(text)
    tokens = [tok.lemma_.lower() for tok in doc
              if not (tok.is_stop or tok.is_punct or (tok.lemma_ == '-PRON-'))]
    return tokens


def build_model():
    """ Build ML pipeline for message classification.
        Model parameter tuning can be found in
        '../model/ML Pipeline Preparation.ipynb'

    Args:
        None
    Returns:
        model (sklearn.pipeline)
    """

    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize, ngram_range=(1, 2))),
        ('tfidf', TfidfTransformer(norm='l1', use_idf=False)),
        ('clf', MultiOutputClassifier(ComplementNB(alpha=0.1)))
    ])
    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    """ Print message classification model evaluation report.
            For each category:
                - Precision
                - Recall
                - F1 score
                - Support

            Summary:
                - Micro avg for each score type
                - Macro avg for each score type
                - Weighted avg for each score type
                - Samples avg for each score type
                - Hamming Loss

    Args:
        model (sklearn.estimator): trained multi-output classifier model
        X_test (pd.DataFrame): test set features
        Y_test (pd.DataFrame): test set targets (all categories)
        category_names ([str]): list of classification category names

    Returns:
        None
    """

    preds = pd.DataFrame(model.predict(X_test), columns=category_names)
    report = classification_report(Y_test, preds,
                                   target_names=category_names)
    print(report)
    print(f'Hamming Loss: {hamming_loss(Y_test, preds)}')


def save_model(model, model_filepath):
    """ Pickle model.

    Args:
        model (sklearn.estimator): classifier model
        model_filepath (str): path to save location

    Returns:
        None
    """

    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


def main():

    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                            test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '
              'as the first argument and the filepath of the pickle file to '
              'save the model to as the second argument. \n\nExample: python '
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()

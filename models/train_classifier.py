# import libraries
import sys
import re
import string
import numpy as np
import pandas as pd
import pickle
from sqlalchemy import create_engine
import nltk
nltk.download(['wordnet', 'punkt', 'stopwords'])
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import  SVC
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier


def load_data(database_filepath):
    """
    Load data from database
    
    Input: sqlite database file
    Output: X feature and y target variables
    """
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table(engine.table_names()[0], engine)
    X = df['message']
    y = df[df.columns[4:]]
    
    return X, y


def tokenize(text):
    """
    Write a tokenization function to process text data
    
    Input: messages
    Output: list of words after processing the following steps
    """
    
    # The following lines of code replace url link with "urlplaceholder"
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    
    # The following lines of code lemmatizes words
    """
    Examples of lemmatization:
    -> rocks : rock
    -> corpora : corpus
    -> better : good
    """
    tokens = word_tokenize(text)
    
    # Remove stopwords
    tokens = [w for w in tokens if w not in stopwords.words("english")]
    
    # Lemmatize words
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        # Convert to lower case and remove whitespace
        clean_tok = lemmatizer.lemmatize(tok).lower().strip() 
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """
    Build a machine learning pipeline take in the message column as input 
    and output classification results on the other 36 categories in the dataset
    
    Output: Best model
    """
    pipeline = Pipeline([
                       ('vect', CountVectorizer(tokenizer=tokenize)),
                       ('tfidf', TfidfTransformer()),
                       ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    parameters = {
            'tfidf__use_idf':[True, False],
            'clf__estimator__n_estimators': [10, 25],
            'clf__estimator__min_samples_split': [3, 4]
        }

    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=2)
    
    return cv


def evaluate_model(model, X_test, Y_test):
    """
    Evaluate model using classification_report
    
    Input: model, X_test, Y_test
    Output: classification_report
    """
    y_pred = model.predict(X_test)
    for i in range(Y_test.shape[1]):
        print(Y_test.columns[i], ':')
        print(classification_report(Y_test.values[:,i], y_pred[:,i]), '_'*50)


def save_model(model, model_filepath):
    # Pickle best model
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y= load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
import sys
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])
import re
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sqlalchemy import create_engine
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
import pickle


def load_data(database_filepath):
    """
        This function loads the data from database into a pandas DataFrame.
        
        input:
        database_filepath: the path to the data base
        
        return:
        X: the features that we will use to train the model
        Y: the target variable 
        Y.columns.values: the name of the cloumns in Y
    """
    engine = create_engine(f"sqlite:///{database_filepath}")
    df = pd.read_sql_table("table_one", con=engine)
    X = df['message']
    Y = df.drop(columns=['id', 'message', 'genre', 'original'])
    
    return X, Y, Y.columns.values

def tokenize(text):
    """
        This function takes a text and normalize and tokenize it then return it. 
    """
    return word_tokenize(re.sub(r"[^a-zA-Z0-9]", " ", text).lower())


def build_model():
    """
        This function return a model that we train and predict on. 
        The model uses pipeline and grid search. 
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    parameters = {
#         'clf__estimator__n_estimators' : [10, 50]
        'clf__estimator__max_features' : ['sqrt', 'log2']
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
        This function takes the model the test set, and the category names.
        
        Then the function predicts on the test set and prints a report for the classifaction of eact category  
    """
    y_pred = model.predict(X_test)
    tmp = pd.DataFrame(y_pred, columns = category_names)
    for i, c in enumerate(category_names):
        print(c)
        print(classification_report(Y_test[c], tmp[c]))
        


def save_model(model, model_filepath):
    """
        This function saves the model as a pickle file. 
    """
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
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
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
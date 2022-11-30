import sys
import re
import pickle
import numpy as np
import pandas as pd

from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import nltk
nltk.download('averaged_perceptron_tagger')
nltk.download(['punkt', 'wordnet'])


def load_data(database_filepath):
    
    '''
    Load the data from the sqlite database
    Args:
        database_filepath (str): database name
    Returns:
        X (df): dataframe containing features
        Y (df): dataframe containing labels
        category_names (list): List of category names
    '''
        
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql('SELECT * FROM disaster_response', con=engine)
    X = df["message"]
    Y = df.iloc[:,3:]
    category_names = list(Y.columns)
    return X, Y, category_names


def tokenize(text):
    
    """
    Prepares the text for the tfidf transformer  
    
    Args:
        text (str): Text message which needs to be tokenized
    Returns:
        clean_tokens (list): List of tokens extracted from the provided text
    """
        
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    
    """
    Starting Verb Extractor class
    
    This class extract the starting verb of a sentence,
    creating a new feature for the ML classifier
    """

    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)
    
    
def build_model():
    
    """
    Set up a ML Pipeline using GridSeach to optimize the parameters  

    Returns:
        model: Optimized ML Pipeline
    """
        
    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),

            ('starting_verb', StartingVerbExtractor())
        ])),

        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    parameters = {
        'clf__estimator__n_estimators': [30, 50],
        'clf__estimator__min_samples_split': [2, 3]
    }

    model = GridSearchCV(pipeline, param_grid=parameters)
    
    return model 


def evaluate_model(model, X_test, Y_test, category_names):
    
    """
    Evaluates the model using the classification report 
    
    Args:
        model: ML pipeline that is used for predicting 
        X_test: Test features 
        Y_test: Test labels
        category_names (list): list with category names 
    """
        
    Y_pred = model.predict(X_test)
    for i in range(len(Y_test.columns)):
        print('Category {}: {} '.format(i, Y_test.columns[i]))
        print(classification_report(Y_test.iloc[:, i].values, Y_pred[:, i]))


def save_model(model, model_filepath):
    
    """
    Saves the trained model into a pickle file for future use  
    
    Args:
        model: fitted ML model 
        model_filepath (string): filepath to save to pickle file to 
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
import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import nltk
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.multioutput import MultiOutputClassifier
import pickle
nltk.download('punkt')
nltk.download('wordnet')


def load_data(database_filepath):
    """
    Load data from SQL database to th ML pipeline
    Input: filepath to the database of interest
    Output: feature df X, target df Y and names of column in Y
    """
    
    engine = create_engine('sqlite:///'+ database_filepath)
    df = pd.read_sql_table('disaster_messages',engine)
    X = df['message'].values
    Y = df.drop(['id','message','original','genre'], axis=1).values
    category_names = df.columns[4:]
    
    return X, Y, category_names


def tokenize(text):
    """
    Clean and tokenize texxt
    Input: text (of the messages)
    Output: cleansed and tokenized text in a list
    """
    
    tokens = nltk.word_tokenize(text)
    lemmatizer = nltk.WordNetLemmatizer()
    lemmatized = [lemmatizer.lemmatize(w).lower().strip() for w in tokens]
    
    return lemmatized


def build_model():
    """
    Builds ML classification pipeline with optimal parameters
    Output: cv object
    """
    pipeline = Pipeline([('cvect', CountVectorizer(tokenizer = tokenize)),
                     ('tfidf', TfidfTransformer()),
                     ('clf', MultiOutputClassifier(RandomForestClassifier()))
                     ])
                     
    parameters = {'clf__estimator__n_estimators': [25, 50],
                         'clf__estimator__min_samples_split': [2,3] 
    }
              
    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=1)
    
    return cv
    
    


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate the performance of the built model
    Input: the trained model, test part of X, test part of Y, names of column in Y
    Outpur: classification report and accuracy score
    """
    Y_pred = model.predict(X_test)
    
        # Print accuracy, precision, recall and f1_score for each categories
    for x in range(0, len(category_names)):
        print(category_names[x])
        print("\tAccuracy: {:.3f}\t\t% Precision: {:.3f}\t\t% Recall: {:.3f}\t\t% F1_score: {:.3f}".format(
            accuracy_score(Y_test[:, x], Y_pred[:, x]),
            precision_score(Y_test[:, x], Y_pred[:, x], average='weighted'),
            recall_score(Y_test[:,x], Y_pred[:, x], average='weighted'),
            f1_score(Y_test[:, x], Y_pred[:, x], average='weighted')))

def save_model(model, model_filepath):
    '''
    Save the model
    Input: model, the file path to where to save the model
    Output: model saved as a pickle file 
    '''
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

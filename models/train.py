from sqlalchemy import create_engine
import pandas as pd
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

def load_data(database):
    """
    Load the data from the sqlite database

    Input:
    - database = path to database file

    Output:
    - X = feature matrix
    - y = response vector
    """

    engine = create_engine('sqlite:///'+database)
    df = pd.read_sql('SELECT * FROM DisasterResponse', engine)
    X = df.iloc[:, 0:4]
    y = df.iloc[:, 4:]

    return X, y


def tokenize(text):
    """
    Tokenizes, lemmatizes and sets all characters to lowercase
    for the input text.

    Input:
    - text = string

    Output:
    - clean_tokens = list of words that have been lemmatized and lowercase
    """

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for token in tokens:
        clean_token = lemmatizer.lemmatize(token).lower().strip()
        clean_tokens.append(clean_token)
    return clean_tokens


def build_model(use_cv=True):
    """
    Build the ML pipeline consisting of the following:
    - CountVectorizer which tokenizes each message
    - TdidfTransfomer which computes the tf-idf values
    - MultiOutputClassifier with RandomForestClassifier

    Input:
    - use_cv = set to True (default) to perform grid search

    Output:
    - model = ML pipeline
    """

    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    if use_cv:
        parameters = {
            'vect__max_features': (5000, 10000),
            'tfidf__use_idf': (True, False),
            'clf__estimator__n_estimators': [100, 200],
        }
        return GridSearchCV(pipeline, param_grid=parameters)

    return pipeline


def display_results(y_test, y_pred):
    """
    Calculate and plot recall, precision, f1-score and accuracy

    Input:
    - y_test = test response vector
    - y_pred = predicted response vector

    Output:
    - None
    """

    # Go through each category, calculate and store the QOIs
    recall    = []
    precision = []
    f1_score  = []
    accuracy  = []
    for i in range(0, 36):
        result = classification_report(y_test.iloc[:, i], [x[i] for x in y_pred], output_dict=True)
        recall.append(result['weighted avg']['recall'])
        precision.append(result['weighted avg']['precision'])
        f1_score.append(result['weighted avg']['f1-score'])
        accuracy.append(result['accuracy'])

    # Sort QOIs for plotting
    recall.sort()
    precision.sort()
    f1_score.sort()
    accuracy.sort()
    plt.plot(recall, label='recall')
    plt.plot(precision, label='precision')
    plt.plot(f1_score, label='f1-score')
    plt.plot(accuracy, label='accuracy')
    plt.legend()

    print('Min recall: {}, Max recall: {}'.format(recall[0], recall[-1]))
    print('Min precision: {}, Max precision: {}'.format(precision[0], precision[-1]))
    print('Min f1_score: {}, Max f1_score: {}'.format(f1_score[0], f1_score[-1]))
    print('Min accuracy: {}, Max accuracy: {}'.format(accuracy[0], accuracy[-1]))


if __name__=="__main__":
    # python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl

    if len(sys.argv) < 3:
        sys.exit("Need at least three arguments")
    database = sys.argv[1]
    if not path.exists(database):
        sys.exit('Database is missing')
    model = sys.argv[2]
    if path.exists(model):
        sys.exit('Model already exists')

    # Load data from database into feature matrix and response vector
    X, y = load_data(database)

    # Split data into train and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    # Build the model
    model = build_model()

    # Fit the model
    model.fit(X_train['message'], y_train)

    # Get the predictions using the model
    y_pred = model.predict(X_test['message'])

    # Display the results
    display_results(y_test, y_pred)

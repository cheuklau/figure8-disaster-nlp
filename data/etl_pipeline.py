import pandas as pd
from sqlalchemy import create_engine
import os.path
from os import path
import sys

def load_data(message_csv, categories_csv, database):
    """
    Loads csv file into a Pandas dataframe and clean it

    Input:
    - message_csv = path to message csv
    - categories_csv = path to categories csv
    - database = path to sqlite database file

    Output:
    - X = features matrix
    - y = response vector
    """

    # Read in the csv files
    messages = pd.read_csv(message_csv, dtype='str', encoding='utf-8')
    categories = pd.read_csv(categories_csv, dtype='str', encoding='utf-8')

    # Merge the dataframes on id
    df = pd.merge(messages, categories, on='id')

    # Split the categories column into separate category columns
    categories = df['categories'].str.split(';', expand=True)

    # Store the category names
    row = categories.iloc[0]
    category_colnames = [x.split('-')[0] for x in row]

    # Convert the categories values to be 0s and 1s
    # The raw data has them pretenfed by the column name
    for column in categories:
        categories[column] = categories[column].str.split('-').str[1].astype(int, copy=False)

    # Rename the categories dataframe column names
    categories.columns = category_colnames

    # Replace the 2's in the related column with 1's
    categories = categories.replace({'related': {2: 1}})

    # Drop the categories column from the dataframe and concatenate the categories dataframe
    df = df.drop('categories', axis=1)
    df = pd.concat([df, categories], axis=1)

    # Drop duplicates but keep the first occurrence
    df = df.drop_duplicates(keep='first')

    # Load to database
    engine = create_engine('sqlite:///'+database)
    df.to_sql('DisasterResponse', engine, index=False)

    # Define feature and response vector
    X = df.iloc[:, 0:4]
    y = df.iloc[:, 4:]

    return X, y


if __name__=="__main__":
    if len(sys.argv) < 3:
        sys.exit('Need at least three arguments')
    message = sys.argv[1]
    if not path.exists(message):
        sys.exit('Message csv missing')
    categories = sys.argv[2]
    if not path.exists(categories):
        sys.exit('Categories csv missing')
    database = sys.argv[3]
    if path.exists(database):
        sys.exit('Database already exists')
    load_data(message, categories, database)

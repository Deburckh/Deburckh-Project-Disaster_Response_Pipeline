import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

# load the data and merge it into one dataframe
def load_data(messages_filepath, categories_filepath):
    
    '''
    Load the data from the input files
    Args:
        categories_filename (str): categories filename
        messages_filename (str): messages filename
    Returns:
        df (pandas.DataFrame): dataframe containing the uncleaned dataset
    '''
    
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    df = messages.merge(categories, on="id")
    
    return df

# clean the categories data and drop duplicates
def clean_data(df):
    
    '''
    Clean the data
    Args:
        df (pandas.DataFrame): dataframe containing the uncleaned dataset
    Returns:
        df (pandas.DataFrame): dataframe containing the cleaned dataset
    '''
    
    categories = df["categories"].str.split(";", expand=True)
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x[:-2])
    categories.columns = category_colnames
    
    for column in categories:
        categories[column] = categories[column].astype(str).str[-1]
        categories[column] = categories[column].astype(int)
        
    categories["related"] = categories["related"].map(lambda x: 1 if x == 2 else x)
        
    df.drop("categories", axis=1, inplace=True)
    df = df.join(categories)
    
    df.drop_duplicates(inplace=True)
    df = df.drop("original", axis=1)
    
    return df

# save the data to a sqlite database 
def save_data(df, database_filepath):
    
    '''
    Save the data into the database. The destination table name is TABLE_NAME
    Args:
        df (pandas.DataFrame): dataframe containing the dataset
        database_filename (str): database filename
    '''
    
    engine = create_engine('sqlite:///' + database_filepath)
    df.to_sql("disaster_response", con=engine, if_exists='replace', index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
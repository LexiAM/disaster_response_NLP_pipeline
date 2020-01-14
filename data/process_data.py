import sys
import pandas as pd
import numpy as np
import sqlite3


def load_data(messages_filepath, categories_filepath):
    """ Load Disaster Response messages and categories data from csv files into
        pandas DataFrame for classification training.

    Args:
        messages_filepath (str): path to messages.csv file
        categories_filepath (str): path to categories.csv file

    Returns:
        df (pd.DataFrame): output frame with combined messages and categories
    """

    messages = pd.read_csv(messages_filepath, na_values=['#NAME?'])
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on='id')
    return df


def clean_data(df):
    """ Clean messages and categories data for message NLP classification.
        - Parse categories column:
          * Split semicolon delimited categories list into individual columns
          * Rename new columns with category names found in first row:
                related-1 | fire-0 | --> related | fire
          * Replace 'category-value' strings with integer values only. Example:
                related-1 --> 1; fire-0 --> 0
        - Drop original column
        - Drop entries with missing messages
        - Recode related values > 1 to 1. Example: related=2 --> related=1
                                                   related=1 --> related=1
        - Drop duplicate values

    Args:
        df (pd.DataFrame): input data

    Returns:
        df_clean (pd.DataFrame): clean data
    """

    assert 'categories' in df, 'KeyError: categories column is not found.'
    assert 'message' in df, 'KeyError: messages column is not found.'

    # split categroies into separate columns
    # input df.categories column: 'related-1;request-0;offer-1; ... '
    # output categories columns: | 'related-1' | 'request-0' | 'offer-1' ...
    categories = df.categories.str.split(';', expand=True)

    # use new categories frame first row to rename columns with category names
    # categories 1st row: | 'related-1' | 'request-0' | 'offer-1 ...
    # new column names: | 'related' | 'request' | 'offer' ...
    categories.columns = categories.loc[0, :].str[:-2].values

    # convert categories 'category-number' values into integers
    # Example: 'related-1' --> 1   'request-0' --> 0
    categories = categories.apply(lambda x: x.str[-1].astype(int))

    # replace df.categories column with the categories dataframe
    df_clean = pd.concat([df.drop(columns='categories'), categories], axis=1)

    # replace blank messages with np.nan
    df_clean.replace(r'^\s*$', np.nan, regex=True, inplace=True)

    # drop 'original' column from df
    df_clean.drop(columns='original', errors='ignore', inplace=True)

    # drop missing values
    df_clean.dropna(inplace=True)

    # drop duplicates
    df_clean.drop_duplicates(subset='message', inplace=True)

    # recode category values greater thaan 1 as 1
    cats = df_clean.loc[:, 'related':]
    df_clean.loc[:, 'related':] = np.where(cats > 1, 1, cats)

    return df_clean


def save_data(df, database_filepath):
    """ Save data for SQLite database in 'message' table.

    Args:
        df (pd.DataFrame): data to be saved
        database_filepath (str): database save path

    Returns:
        None
    """

    # create database connection using sqlite3
    conn = sqlite3.connect(database_filepath)

    # drop message table if it already exists -- override old data
    cur = conn.cursor()
    cur.execute('DROP TABLE IF EXISTS message')

    # save data to database table `message`
    df.to_sql('message', conn, index=False)


def main():

    if len(sys.argv) == 4:

        (messages_filepath, categories_filepath,
         database_filepath) = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '
              'datasets as the first and second argument respectively, as '
              'well as the filepath of the database to save the cleaned data '
              'to as the third argument. \n\nExample: python process_data.py '
              'disaster_messages.csv disaster_categories.csv '
              'DisasterResponse.db')


if __name__ == '__main__':
    main()

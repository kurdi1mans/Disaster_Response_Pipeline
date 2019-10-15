import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    return messages,categories


def clean_data(df,cat_df):

    # parsing categories to seperate columns
    expanded_categories = cat_df['categories'].str.split(";",expand=True)
    
    # extract category names from the first row
    row = expanded_categories.loc[0,:].astype(str)
    category_colnames = row.str.slice(0,-2,1).values

    # assign column names to the category columns
    expanded_categories.columns = category_colnames

    for column in expanded_categories:
      # set each value to be the last character of the string
      cat_df[column] = expanded_categories[column].str[-1:]
      
      # convert column from string to numeric
      cat_df[column] = cat_df[column].astype(int)

    # drop the original 'categories' column which holds the raw data.
    cat_df.drop(['categories'],axis=1,inplace=True)

    # merge the categories table with the messages table into a single table
    df = df.merge(cat_df,left_on="id",right_on="id")

    # remove duplicates based on repeated messages
    df.drop_duplicates(subset=['message'],inplace=True)
    return df

def save_data(df, database_filename):
    engine = create_engine('sqlite:///'+database_filename)
    
    # store the DataFrame 'df' into the table 'message'
    df.to_sql('message', engine, index=False) 


def main():
    if len(sys.argv) == 4:
        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]
        
        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df,categories = load_data(messages_filepath, categories_filepath)


        print('Cleaning data...')
        df = clean_data(df,categories)

        
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
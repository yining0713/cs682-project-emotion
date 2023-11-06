# Split the test set from the original data
# Only train on these classes: Worry, happiness, sadness, relief, hate, boredom
# Select 200 samples from each class as the test set;
# For boredom, because there are originally only 179 samples, only get 30 samples for testing
# basic cleaning is done on the dataset
import pandas as pd
import numpy as np
import re
import sys
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

CLASSES = ["worry", "happiness", "sadness", "relief", "hate", "boredom"]
LIMITED_CLASSES = ["boredom"]
OUTPUT_TRAIN = "../datasets/clean_data_train.csv"
OUTPUT_TEST = "../datasets/clean_data_test.csv"

def read_data(input_file):
        data = pd.read_csv(input_file)
        data = data[data['content'] != ''] # Deleting Empty Lines
        return data

def choose_classes_and_split_train_test(dataframe):
    train = []
    test = []
    ntest = 200 # number of tests
    for sentiment in CLASSES:
        subset_all = dataframe.loc[dataframe["sentiment"]==sentiment]
        if sentiment in LIMITED_CLASSES: # if the original sample is too small, extract only a few for testing
            ntest = 30
        subset_test = subset_all.sample(n=ntest) # Randomly extract tests
        subset_train = subset_all.drop(subset_test.index) # The rest is training data
        train.append(subset_train)
        test.append(subset_test)
    return pd.concat(train), pd.concat(test)

def clean_basic(text):
    # Removing URLs
    text = re.sub(r'http\S+', '', text)
    # Removing User Mentions and Hashtag symbols
    text = re.sub(r'[@#][\w]*', '', text)
    # Removing RT（Retweet）
    text = re.sub(r'RT', '', text)
    return text

def additional_clean(text):
    # Removing Special Characters and Punctuation
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    # Removing numbers from tweet text
    text = re.sub(r'\d', '', text)
    # Converting to lowercase
    text = text.lower()
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words]
    # Removing Stop Words
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

def clean_dataframe(dataframe):
    dataframe['cleaned_content'] = dataframe['content'].apply(clean_basic)
    dataframe['super_cleaned_content'] = dataframe['cleaned_content'].apply(additional_clean)
    return dataframe

def write_output(dataframe, output):
    dataframe[['sentiment', 'cleaned_content', 'super_cleaned_content', 'content']].to_csv(output)
    return

def main():
    input = sys.argv[1]
    df_from_input = read_data(input)
    cleaned_df = clean_dataframe(df_from_input)
    df_train, df_test = choose_classes_and_split_train_test(cleaned_df)
    write_output(df_train, OUTPUT_TRAIN)
    write_output(df_test, OUTPUT_TEST)
    # print(type(df_train))

if __name__ == "__main__":
    main()

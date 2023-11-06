# CS682 Project: Using LLM to augment text data for emotion detection

## Dataset
- https://www.kaggle.com/datasets/pashupatigupta/emotion-detection-from-text/

## Dataset cleaning
- Basic cleaning: remove irrelevant texts in tweets, like "RT", website URLs, and "@username"
- Cleaning for the purpose of traditional feature extraction methods:
    - remove special characters and punctuations
    - lowercase
    - lemmatize
    - remove stop words

## Dataset split
- Choosing classes: only choosing "worry", "happiness", "sadness", "relief", "hate", "boredom" for this experiment
- 200 samples are randomly selected from each class for testing, the rest is for training or validation. The class "boredom" only has 179 samples to begin with, so we are only taking 30 samples for testing.

## Files
- Training data: datasets/clean_data_train.csv
- Testing data: datasets/clean_data_test.csv
- dataset format: 
    - four columns: sentiment, cleaned_content, super_cleaned_content(for more traditional feature extraction), content(original)


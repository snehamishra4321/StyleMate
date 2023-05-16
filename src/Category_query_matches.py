import pandas as pd
from fuzzywuzzy import fuzz
import string
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from textblob import TextBlob
from nltk.stem import WordNetLemmatizer
from collections import Counter
from helpers import get_data_from_file
import pdb

# Define a function to preprocess the query
def preprocess_query(query):
    # Convert to lowercase
    query = query.lower()
    # Remove punctuation
    query = query.translate(str.maketrans("", "", string.punctuation))
    # Tokenize
    tokens = query.split()
    # Remove stop words
    stop_words = set(stopwords.words("english"))
    tokens = [token for token in tokens if token not in stop_words]
    # Spell check
    #spell = TextBlob("")
    #tokens = [spell.correct(token) for token in tokens]
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    # Join tokens back into a string
    query = " ".join(tokens)
    return query


# Define a function that returns a list of all the ASIN ids associated with the top matching categories.
def find_matching_categories_fuzzy(query, category_dict_file_path, n=5):
    # Preprocess the query
    query = preprocess_query(query)
    # Tokenize the query
    query_tokens = set(query.split())
    #Load the category_dict
    category_dict = get_data_from_file(category_dict_file_path)
    # Find matching categories
    matching_categories = []
    for category, asins in category_dict.items():
        category_tokens = set(category.split())
        overlap = query_tokens.intersection(category_tokens)
        if len(overlap) > 0 or fuzz.token_set_ratio(query, category) >= 70:
            matching_categories.append((category, asins))
    # Sort matching categories by their fuzzy string match score
    matching_categories = sorted(matching_categories, key=lambda x: fuzz.token_set_ratio(query, x[0]), reverse=True)
    # Return top n matching categories
    top_categories = matching_categories[:n]
    matching_indices = []
    for category, asins in top_categories:
        matching_indices.extend(asins)
    return matching_indices


if __name__  == '__main__':

    query = "shirt"
    category_dict_file_path = './Output/category_dict_.pickle'
    print(find_matching_categories_fuzzy(query=query, category_dict_file_path=category_dict_file_path, n=10))
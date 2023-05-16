from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from spellchecker import SpellChecker

import pickle


# Run the below code if you are running for the first time run below code
# nltk.download()




def text_preprocessing(text: str):
    
    # Lowercase
    text = text.lower()
    # Remove Punctuation
    # text = "".join([char for char in text if char not in string.punctuation])
    words = word_tokenize(text)  
    # Remove Stopwords
    stop_words = stopwords.words('english')
    filtered_words = [word for word in words if word not in stop_words]
    ## Stemming
    # porter = PorterStemmer()
    # stemmed = [porter.stem(word) for word in filtered_words]
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    lemmatized = [lemmatizer.lemmatize(word) for word in filtered_words]
    
    return ' '.join(lemmatized)

def get_data_from_file(filename:str):

    with open(filename, 'rb') as file:
        data = pickle.load(file)
    return data


def get_top_n_indices(array, top_n):
    return array.argsort()[-top_n:][::-1]


def query_preprocessing(text: str):

    # Lowercase
    text = text.lower()
    # Remove Punctuation
    # text = "".join([char for char in text if char not in string.punctuation])
    words = word_tokenize(text)  
    # Remove Stopwords
    # stop_words = stopwords.words('english')
    # filtered_words = [word for word in words if word not in stop_words]
    filtered_words = [spell.correction(word) in words]
    ## Stemming
    # porter = PorterStemmer()
    # stemmed = [porter.stem(word) for word in filtered_words]
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    lemmatized = [lemmatizer.lemmatize(word) for word in filtered_words]
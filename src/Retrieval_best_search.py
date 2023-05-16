#!/usr/bin/env python
# coding: utf-8

# In[112]:


import pandas as pd

df = pd.read_csv('Metadata.csv')


# In[113]:


df.head().iloc[0]


# In[114]:


for index, row in df.tail(30).iterrows():
    print(row['category'])


# In[115]:


if df['category'].str.contains('Clothing, Shoes & Jewelry').all():
    print("All rows contain 'Clothing, Shoes & Jewelry' in the 'category' column.")
else:
    print("Not all rows contain 'Clothing, Shoes & Jewelry' in the 'category' column.")


# In[116]:


count = 0
for cat in df['category']:
    if 'Clothing, Shoes & Jewelry' in cat:
        count += 1
print(count)


# In[117]:


if df['category'].str.lower().str.contains('men|women').all():
    print("All rows contain 'Men' or 'Women' in the 'category' column.")
else:
    print("Not all rows contain 'Men' or 'Women' in the 'category' column.")


# In[118]:


count = 0
for cat in df['category']:
    if 'Boy' in cat or 'Girl' in cat or 'Boys' in cat or 'Girls' in cat:
        count += 1
print(f"Number of rows with 'Girl/Girls' or 'Boy/Boys' in the category column: {count}")


# In[119]:


count = 0
for cat in df['category']:
    if 'Men' in cat or 'Women' in cat:
        count += 1
print(f"Number of rows with 'Men' or 'Women' in the category column: {count}")


# In[120]:


df['category'] = df['category'].apply(lambda x: [s.strip() for s in eval(x)])
#df['category'] = df['category'].apply(lambda x: [cat.replace('Clothing, Shoes & Jewelry', '').strip() for cat in x])
df['category'] = df['category'].apply(lambda x: [cat.strip() for cat in x if cat.strip() != 'Clothing, Shoes & Jewelry'])


# In[121]:


df['category'].head()


# In[122]:


category_dict = {}

for i, row in df.iterrows():
    categories = row['category']
    asin = row['asin']
    for category in categories:
        category = category.strip().lower()
        if len(category.split()) <= 4:
            if category not in category_dict:
                category_dict[category] = [asin]
            else:
                category_dict[category].append(asin)
        else:
            continue


# In[123]:


for key, value in list(category_dict.items())[:5]:
    print(f"{key}: {value}")


# In[148]:


unique_categories_count = len(category_dict.keys())
print(f"Number of unique categories: {unique_categories_count}")


# In[149]:


for key, value in category_dict.items():
    print(f"{key}")


# In[174]:


import string
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


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
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    # Join tokens back into a string
    query = " ".join(tokens)
    return query


# In[183]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# Define a function to find the best matching categories for a query
def find_matching_categories(query, category_dict, n=15):
    # Preprocess the query
    query = preprocess_query(query)
    # Vectorize the categories and query
    categories = list(category_dict.keys())
    vectorizer = CountVectorizer().fit_transform([query] + categories)
    vectors = vectorizer.toarray()
    query_vector = vectors[0].reshape(1, -1)
    category_vectors = vectors[1:]
    # Calculate cosine similarity between the query and categories
    similarity_scores = cosine_similarity(query_vector, category_vectors)[0]
    # Find the top matching categories
    top_indices = np.argsort(similarity_scores)[::-1][:n]
    top_categories = [(categories[i], category_dict[categories[i]]) for i in top_indices]
    return top_categories


# In[184]:


query = "red dress"
matching_categories = find_matching_categories(query, category_dict)
print(matching_categories)


# In[181]:


def find_matching_categories(query, category_dict, n=10):
    # Preprocess the query
    query = preprocess_query(query)
    
    # Vectorize the categories and query using TF-IDF
    categories = list(category_dict.keys())
    vectorizer = TfidfVectorizer().fit_transform([query] + categories)
    vectors = vectorizer.toarray()
    query_vector = vectors[0].reshape(1, -1)
    category_vectors = vectors[1:]
    
    # Compute cosine similarity between the query vector and category vectors
    similarities = cosine_similarity(query_vector, category_vectors).flatten()
    
    # Sort categories by descending similarity and return the top n
    sorted_indices = similarities.argsort()[::-1]
    top_indices = sorted_indices[:n]
    matching_categories = [(categories[i], category_dict[categories[i]]) for i in top_indices]
    
    return matching_categories


# In[182]:


query = "red dress"
matching_categories = find_matching_categories(query, category_dict)
print(matching_categories)


# In[ ]:





# In[ ]:





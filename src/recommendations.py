import pandas as pd
import numpy as np
import pdb
from tqdm import tqdm
import nltk
import string
import pickle 
import yaml
import os
import torch 
import pdb 

from helpers import text_preprocessing, get_data_from_file, get_top_n_indices
from category_query_matches import find_matching_categories_fuzzy
from sentence_transformers import SentenceTransformer


CONFIG_FILE = 'config.yaml'


class Recommender():

    def __init__(self, primary_column:str = 'title', top_n:int = 10) -> None:

        # Open the configuration file to load parameters
        with open(CONFIG_FILE, "r") as file:
            try:
                self.params = yaml.safe_load(file)
            except yaml.YAMLError as exc:
                print(exc)
        
        self.top_n = top_n
        self.primary_column = primary_column
        metadata_file = self.params['METADATA_FILE']
        self.metadata = pd.read_csv(metadata_file, index_col=0)
        self.preprocess_metadata()
        print("INFO: Initializing Model")
        self.model = SentenceTransformer('msmarco-distilbert-base-dot-prod-v3')
        self.embeddings = self.get_embeddings(column=primary_column)
        self.category_dict_path = self.params['CATEGORY_DICT_PATH']

        # self.product_feature_positiveness = get_data_from_file(self.params['product_feature_ratings'])
        self.image_similar_product_dict = get_data_from_file(self.params['IMAGE_SIMILARITY_DICT'])


    def preprocess_metadata(self) -> None:

        self.metadata['description'] = self.metadata['description'].apply(lambda x: text_preprocessing(eval(x)[0]))

    def get_embeddings(self, column) -> np.array:

        # If embedding is locally saved already, load it 
        try:
            with open(self.params['EMBEDDING_FILE'], 'rb') as file:
                embeddings = pickle.load(file)
            print("INFO : Loaded Product Embeddings")
            return embeddings
        
        # If embedding is not locally available, create embeddings
        except:
            print("INFO : Creating Embeddings")
            if torch.cuda.is_available():
                embeddings = self.model.encode(self.metadata[column].tolist(), device='cuda') 
            else:
                embeddings = self.model.encode(self.metadata[column].tolist())
            embeddings = np.asarray(embeddings.astype('float32'))   
            
            print("INFO: Saving embeddings")
            if not os.path.exists(os.path.dirname(self.params['EMBEDDING_FILE'])):
                os.mkdir(os.path.dirname(self.params['EMBEDDING_FILE']))
            with open(self.params['EMBEDDING_FILE'],'wb') as file:
                pickle.dump(embeddings, file, protocol=pickle.HIGHEST_PROTOCOL)

            return embeddings

    def return_most_similar(self, query):

        print("INFO: Retrieving items for query")
        query_vector = self.model.encode([query])
        similarity = np.dot(self.embeddings,query_vector.T)
        top_items = similarity.flatten().argsort()[-self.top_n:][::-1]
        print(self.metadata['title'].iloc[top_items])
        # return list(top_items), list(self.metadata['title'].iloc[top_items])
        return list(top_items), list(self.metadata['asin'].iloc[top_items])


    def character_similarity(self, character_list:list, method:int = 1):

        # Method 1 - Join all characteristics to make an expanded query
        if method == 1 :
            character_query = ' '.join(character_list)
            character_query_vector = self.model.encode(character_query)
            similarity = np.dot(self.embeddings,character_query_vector.T)
            # top_items =  get_top_n_indices(similarity.flatten(), self.top_n)       
            return similarity

        # Method 2 - Rank items based on individual characteristics
        elif method == 2:
            similarity_list =[]
            for characteristic in character_list:
                character_query_vector = self.model.encode(character_query)
                similarity = np.dot(self.embeddings,character_query_vector.T)
                # top_items_list.append(get_top_n_indices(similarity.flatten(), top_n=200)) 
                similarity_list.append(similarity.flatten()) 
            agg_similarity = np.array(similarity_list.sum(axis=0))
            return agg_similarity
        
    def feature_similarity(self, feature_imp):

        agg_feature_poitiveness = np.zeros(self.product_feature_positiveness.shape()[0])
        for ind, imp in enumerate(feature_imp):
            agg_feature_poitiveness += imp*self.product_feature_positiveness[ind]
        return agg_feature_poitiveness


    def return_most_similar_v1(self, query:str, character_list:list, character_method:int, feature_imp:list = None):

        print("INFO: Retrieving items for query")
        filtered_asins = find_matching_categories_fuzzy(
            query=query, category_dict_file_path=self.category_dict_path, n=10)
        
        asin_ind_mapping = get_data_from_file(self.params['ASIN_IND_MAPPING_DICT'])
        filtered_indices = [asin_ind_mapping[asin] for asin in filtered_asins]
        # filtered_indices = [i for i in range(5550)]
        query_vector = self.model.encode([query + ' '.join(character_list)])
        # pdb.set_trace()
        query_similarity = np.dot(self.embeddings[filtered_indices],query_vector.T).flatten()
        query_similarity  /= np.linalg.norm(query_similarity)
        final_similarity = query_similarity
        
        top_filtered_indices = get_top_n_indices(final_similarity, 10)
        indices = [filtered_indices[int(i)] for i in top_filtered_indices]
        # pdb.set_trace()
        # feature_similarity = self.feature_similarity(feature_imp)
        # return indices
        return self.metadata['title'].iloc[indices].to_list()
                 
    def get_image_based_similar_items(self, product_asin:str):

            return self.image_similar_product_dict[product_asin]

    # def get_top_items_for_features(top_n):
        
    #     self
    #     final_product_embeddings = self.product_feature_ratings.mean(axis=1)
    #     top_item_ind = get_top_n_indices(final_product_embeddings, top_n=5)
    #     return top_item_ind



if __name__ == '__main__':

    query = "shoes"
    character_list = ['Loose', 'cotton']
    recommender = Recommender(primary_column='description')
    # print(recommender.return_most_similar_v1(query=query, character_list=character_list, character_method=1))   
    print(recommender.get_image_based_similar_items(product_asin='B0027WH4YO'))

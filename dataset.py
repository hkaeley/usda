import pandas as pd #We do dataprocessing with pandas because the dataset is so big.
import re
import sklearn
from sklearn.model_selection import train_test_split
import pickle as pkl


class Dataset():
    def __init__(self, filename = "/content/dsusda/branded_food.csv", nutrients_file = "/content/dsusda/food_nutrient.csv", load_dataset = False):
        if load_dataset == False:
            self.create_data(filename = filename, nutrients_file = nutrients_file)

    def create_data(self, filename = "/content/dsusda/branded_food.csv", nutrients_file = "/content/dsusda/food_nutrient.csv"):
                #The following cell splits data into
        #We create training and validation sets 
        
        m = 400000
        for i, chunk in enumerate(pd.read_csv(filename, chunksize=m)):
            if i == 0:
                X_tr = chunk[["fdc_id", "ingredients","branded_food_category"]] 
                break
        #This cell finds the nutrient ids for specific labels
        
        
        Y_tr = pd.read_csv(nutrients_file)

        nutrients_of_interest = {1003, 1004, 1005, 1008}
        Y_tr_interest = Y_tr[Y_tr["nutrient_id"].isin(nutrients_of_interest)][["fdc_id","nutrient_id","amount"]]


        Ytr = Y_tr_interest.pivot_table(index='fdc_id', values='amount', columns='nutrient_id', aggfunc='sum', fill_value=0)


        X_tr["ingredients"] = X_tr["ingredients"].astype(str)

        #To create the proper Xtr, we need to find the top n popular ingredients

        X_tr['ingredients'] = X_tr['ingredients'].str.lower().str.replace('ingredients:','').str.split(',').apply(
            lambda ings: set([re.sub(r"[^a-zA-Z0-9\s]+", "", ing.strip()) for ing in ings]))

        X_tr_per_ingredient = X_tr.explode('ingredients')
        ingredient_counts = X_tr_per_ingredient['ingredients'].value_counts()
        num_ingredients = 100
        top_n_ingredients = ingredient_counts[:num_ingredients]

        X_tr_top = X_tr_per_ingredient[X_tr_per_ingredient['ingredients'].isin(top_n_ingredients.keys())]

        print(X_tr_top.shape)

        print(X_tr_top[['fdc_id','ingredients']].shape)

        Xtr = X_tr_top[['fdc_id','ingredients']].pivot_table(index='fdc_id', columns='ingredients', aggfunc=len, fill_value=0)

        categories = X_tr_top[['fdc_id','branded_food_category']].groupby('fdc_id').last()

        #Make sure both datasets only include the data they share. 
        merged_tr = pd.merge(Xtr, Ytr, left_index=True, right_index=True)


        self.y = merged_tr[nutrients_of_interest].to_numpy()
        self.x = merged_tr[Xtr.columns].to_numpy()

    def split_dataset(self):
        return sklearn.model_selection.train_test_split(self.x, self.y, train_size = 0.7, random_state = 42) #split into train and test

    def save(self, dataset_save_path):
        with open(dataset_save_path, 'wb') as f:
            pkl.dump(self, f)

    def load(self, dataset_load_path):
        with open(dataset_load_path, 'rb') as f:
            return pkl.load(f)

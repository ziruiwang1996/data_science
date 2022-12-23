# Implement a movie rating prediction program for both item-based and user-based collaborative
# filtering for the MovieLens 100k dataset available at http://grouplens.org/datasets/movielens/
# inputs: a user-item matrix (same format as u.data),
#         the neighborhood size,
#         a user id ’u’,
#         an item-id ’i’
# output: predicted rating for both user-based CF and item-based CF as output.

import pandas as pd
import numpy as np
from numpy.linalg import norm

class CollaborativeFiltering:
    def __init__(self, neighborhoodSize, userId, itemId, totalUsers, totalItems, udata):
        self.neighborhoodSize = neighborhoodSize
        self.u = userId-1
        self.i = itemId-1
        self.totu = totalUsers
        self.toti = totalItems
        self.user_item_matrix = udata

    def get_user_pair_array(self, a_index, b_index):
        users_array = self.user_item_matrix[[a_index, b_index],:]
        # remove columns with any NaN
        users_array = users_array[:, ~np.isnan(users_array).any(axis=0)]
        user_a = users_array[0]
        user_b = users_array[1]
        return user_a, user_b

    def get_item_pair_array(self, a_index, b_index):
        items_array = self.user_item_matrix[:,[a_index, b_index]]
        # remove rows with any NaN
        items_array = items_array[~np.isnan(items_array).any(axis=1), :]
        items_array = np.transpose(items_array)
        item_a = items_array[0]
        item_b = items_array[1]
        return item_a, item_b

    def cosine_similarity(self, a, b):
        if len(a) == 0 or len(b) == 0:
            return 0
        return np.dot(a, b)/(norm(a)*norm(b))

    def find_user_neighbors(self):
        similar_users_dict = {}
        for user in range(self.totu):
            if user == self.u :
                continue
            user_a_b = self.get_user_pair_array(self.u, user)
            user_a = user_a_b[0]
            user_b = user_a_b[1]
            sim = self.cosine_similarity(user_a, user_b)
            similar_users_dict[user] = sim
        sorted_users = sorted(similar_users_dict.items(), key = lambda x:x[1])
        selected_users = sorted_users[-self.neighborhoodSize:]
        return dict(selected_users)

    def find_item_neighbors(self):
        similar_items_dict = {}
        for item in range(self.toti):
            if item == self.i :
                continue
            item_a_b = self.get_item_pair_array(self.i, item)
            item_a = item_a_b[0]
            item_b = item_a_b[1]
            sim = self.cosine_similarity(item_a, item_b)
            similar_items_dict[item] = sim
        sorted_items = sorted(similar_items_dict.items(), key = lambda x:x[1])
        selected_items = sorted_items[-self.neighborhoodSize:]
        return dict(selected_items)

    def get_avg(self, array):
        return np.nanmean(array)

    def userBasedRatingPrediction(self):
        numerator = 0
        denominator = 0
        for user, sim in self.find_user_neighbors().items():
            user_avg = self.get_avg(self.user_item_matrix[user, :])
            # check if the rate is missing
            if np.isnan(self.user_item_matrix[user, self.i]):
                continue
            else:
                value = sim * (self.user_item_matrix[user, self.i] - user_avg)
                numerator = numerator + value
                denominator = denominator + sim
        if denominator == 0 :
            return "Fail to predict due to too many missing values"
        else:
            rating = self.get_avg(self.user_item_matrix[self.u, :]) + numerator / denominator
            return round(rating, 2)

    def itemBasedRatingPrediction(self):
        numerator = 0
        denominator = 0
        for item, sim in self.find_item_neighbors().items():
            item_avg = self.get_avg(self.user_item_matrix[:, item])
            if np.isnan(self.user_item_matrix[self.u, item]):
                continue
            else:
                value = sim * (self.user_item_matrix[self.u, item] - item_avg)
                numerator = numerator + value
                denominator = denominator + sim
        if denominator == 0 :
            return "Fail to predict due to too many missing values"
        else:
            rating = self.get_avg(self.user_item_matrix[:, self.i]) + numerator / denominator
            return round(rating, 2)

# data preprocessing: generate user-item matrix
matrix = np.empty([943, 1682])
matrix[:] = np.nan
with open("ml-100k/u.data") as input:
    ratings = input.readlines()
for each in ratings:
    each = each.strip().split("\t")
    user, item, rating = int(each[0]), int(each[1]), int(each[2])
    matrix[user-1, item-1] = rating

# predictions
model = CollaborativeFiltering(neighborhoodSize=100, userId=50, itemId=50, totalUsers=943, totalItems=1682, udata=matrix)
print("User-based rating for user ", model.u+1, "toward movie", model.i+1, ":", model.userBasedRatingPrediction())
print("Item-based rating for user ", model.u+1, "toward movie", model.i+1, ":", model.itemBasedRatingPrediction())

# Terminal Output:
# User-based rating for user  50 toward movie 50 : 4.14
# Item-based rating for user  50 toward movie 50 : Fail to predict due to too many missing values

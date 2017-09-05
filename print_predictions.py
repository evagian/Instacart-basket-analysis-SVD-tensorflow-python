import numpy as np # linear algebra
import pprint
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import csv

# Pre
# input: predicted basket size of the next order of every user
# output: 2 lists of lists
# 1st list: the products that each user actually bought in his next order
# 2nd list: prediction of the products that each user will buy in his next order


orders = pd.read_csv("data/orders_train_test.csv")
prior = pd.read_csv("data/order_products__prior.csv")
train = pd.read_csv("data/order_products__train.csv")

# concatenate prior and train orders in one dataframe
frames = [prior, train]
products = result = pd.concat(frames)

# only keep the orders that belong in the test set
# for every user, we want to predict the products of his last basket

# ['group_indicator'] == 1, indicates that the specified order belongs to the test set
orders = orders.loc[orders['group_indicator'] == 1]

# find the products that were actually bought in the last order of every user
test_orders = pd.merge(orders, products, on='order_id')
print test_orders.head(10)

test_orders2 = test_orders[['user_id', 'order_id', 'product_id']]
print test_orders2.head(10)

# create a list of lists
# list of the last order of each user
# containing lists of the products that each user bought in his last order
test_orders2 =  test_orders2.groupby(['user_id', 'order_id'])['product_id'].apply(list)
print test_orders2.head(10)
#filename = 'actual_products.csv'
#test_orders2.to_csv(filename, index=False, encoding='utf-8', header=False)


test_set = pd.read_csv("data/test_set_.csv", names = ["user_id", "days_since_prior_order", "basket", "order_id"])
# the next dataset contains the predicted basket size of the next basket (output of svd_train_val.py)
preds = pd.read_csv("data/pred-actual.csv",  names = ["pred", "actual"])
# this dataset contains statistics concerning users' consumer behaviour
user_prod_stats = pd.read_csv("data/user_product_stats.csv")
#act_prods = pd.read_csv("data/actual_products.csv")


test_preds = pd.concat([test_set, preds], axis=1)

pred_prods =pd.DataFrame()
l=int(len(test_set))
c=int(1)
final_pred_prods = []
final_pred_prods2 = pd.DataFrame()

i = 0

# iterate through the dataframe containing the user_id and the predicted number of his next basket

# for every user check the predicted size of his next basket and accordingly predict the products that he buy

# the prediction of the next basket products depends on the following:
# 1. predicted basket size
# 2. the preferences of the user,
for index, row in test_preds.iterrows():
     user_stats = []
     basket_size = int(round(row['pred'],0))
     user = row['user_id']

     user_stats = user_prod_stats.loc[user_prod_stats['user_id'] == user]
     user_products = user_stats['product_id']

     pred_prods  =  user_products.head(basket_size)
     df_row = pred_prods.tolist()
     final_pred_prods.append(df_row)

     i = i+1

print type(final_pred_prods)
print type(final_pred_prods[1])

print 'results'
for xs in final_pred_prods:
    print ",".join(map(str, xs))

# create a list of lists
# list of the last order of each user
# containing lists of the predicted products for this order
'''
with open('data/pred_products.csv', 'wb') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    #wr = ",".join(map(str, wr))
    wr.writerow(final_pred_prods)
'''



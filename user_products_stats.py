import numpy as np # linear algebra
import pprint
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# calculate user consumer behaviour statistics

order_products_prior_df = pd.read_csv("data/order_products__prior.csv")
orders_df = pd.read_csv("data/orders.csv")

print 'Merge products with prior orders'
order_products_prior_df = pd.merge(order_products_prior_df, orders_df, on='order_id')
order_products_prior_df = order_products_prior_df.drop_duplicates()

# 1 - calculate the percent of baskets in which user x bought product y

number_of_orders = order_products_prior_df.groupby(['user_id','order_id']).size().reset_index(name="order_times")
number_of_orders = number_of_orders.groupby(['user_id']).size().reset_index(name="order_times")

products_bought = order_products_prior_df.groupby(['user_id','product_id']).size().reset_index(name="product_times")

users_products = pd.merge(number_of_orders, products_bought, on='user_id')
users_products['prod_percent_baskets'] = users_products['product_times']/users_products['order_times']

users_products_percent = users_products.drop(['order_times', 'product_times'], axis=1)
print users_products.head(5)
#filename = 'data/users_products_percent_baskets.csv'
#users_products_percent.to_csv(filename, index=False, encoding='utf-8')


# 2 - calculate how often (days_since_prior_order) does each user buy each product

days_since_prior_order = order_products_prior_df.groupby(['user_id', 'product_id'])['days_since_prior_order'].sum().reset_index(name='days_since_prior_order')

product_buys = order_products_prior_df.groupby(['user_id', 'product_id']).size().reset_index(name='number')

days_since_prior_order = pd.merge(days_since_prior_order, product_buys, on=['user_id', 'product_id'])


days_since_prior_order['days_since_prior_order2'] = days_since_prior_order['days_since_prior_order']/days_since_prior_order['number']

print days_since_prior_order.head(5)
#filename = 'data/days_since_prior_order.csv'
#days_since_prior_order.to_csv(filename, index=False, encoding='utf-8')

#########################

# 3 - calculate the average order in which each user buys each product
# for example one user buys 1 package of eggs, 1 kilo apples and then 1 package of flour
# the order number of the eggs = 1
# the order number of the apples = 2
# the order number of the flour = 3

# users tend to rebuy products that appear high in the order position
add_to_cart_order = order_products_prior_df.groupby(['user_id', 'product_id'])['add_to_cart_order'].sum().reset_index(name='order')

product_times = order_products_prior_df.groupby(['user_id', 'product_id']).size().reset_index(name='number')

add_to_cart_order = pd.merge(add_to_cart_order, product_times, on=['user_id', 'product_id'])

add_to_cart_order['average_order_position'] = add_to_cart_order['order']/add_to_cart_order['number']

print add_to_cart_order.head(5)
#filename = 'data/average_order_position.csv'
#add_to_cart_order.to_csv(filename, index=False, encoding='utf-8')

#########################

# 4 - calculate how often does each user rebuy each product

number_of_reorders = order_products_prior_df.groupby(['user_id', 'product_id'])['reordered'].sum().reset_index(name='order')

product_orders = order_products_prior_df.groupby(['user_id', 'product_id']).size().reset_index(name='number')

number_of_reorders = pd.merge(number_of_reorders, product_orders, on=['user_id', 'product_id'])

number_of_reorders['percent_reordered'] = number_of_reorders['order']/number_of_reorders['number']

print number_of_reorders.head(5)
#filename = 'data/users_products_reorder_percent_baskets.csv'
#number_of_reorders.to_csv(filename, index=False, encoding='utf-8')

#########################

# concatenate all user-product metrics into a new dataframe

#day = pd.read_csv("data/days_since_prior_order.csv")
#position = pd.read_csv("data/average_order_position.csv")
#reorder = pd.read_csv("data/users_products_reorder_percent_baskets.csv")
#order = pd.read_csv("data/users_products_percent_baskets_etc.csv")
user_product_stats = pd.merge(users_products_percent, number_of_reorders, on=['user_id', 'product_id'])
user_product_stats = pd.merge(user_product_stats, add_to_cart_order, on=['user_id', 'product_id'])
user_product_stats = pd.merge(user_product_stats, days_since_prior_order, on=['user_id', 'product_id'])
user_product_stats = user_product_stats[['user_id','product_id',  'prod_percent_baskets',  'percent_reordered', 'average_order_position', 'days_since_prior_order2' ]]
print user_product_stats.head(10)

user_product_stats = user_product_stats.sort_values(['user_id', 'prod_percent_baskets', 'average_order_position', 'percent_reordered', 'days_since_prior_order2'], ascending=[True, False, True, False, True])
print user_product_stats.head(30)

filename = 'data/user_product_stats.csv'
user_product_stats.to_csv(filename, index=False, encoding='utf-8')

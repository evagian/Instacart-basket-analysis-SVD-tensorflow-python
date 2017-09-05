Description 

The dataset used for this project purposes consists of 3 million open source online grocery store orders from more than 200 thousands of users. The dataset was available in one of the Kaggle’s competitions named ‘Instacart Market Basket Analysis’. This competition challenged data miners from all over the world to answer to the following question: “Which products will an Instacart consumer purchase in his next basket?”. 

Market basket analysis is an important component of every retail company. Simple, yet powerful - MBA is an inexpensive technique to identify cross-sell opportunities and engage customers. At the same time, personalized recommendation systems differentiate companies from the competition and they can lead to competitive advantages. Moreover, recommendation systems are proven to improve user experience, to increase user traffic and the number of purchases and to encourage user engagement and satisfaction. This competition was an opportunity for us to expand our knowledge and to gain hands on experience on models and techniques used in the fields of basket analysis and recommendation systems.


```python
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

color = sns.color_palette()

%matplotlib inline

pd.options.mode.chained_assignment = None  # default='warn
```


```python
from subprocess import check_output
print(check_output(["ls", "data"]).decode("utf8"))
```

    aisles.csv
    departments.csv
    order_products__prior.csv
    order_products__train.csv
    orders.csv
    products.csv
    sample_submission.csv
    Untitled Folder
    



```python
order_products_train_df = pd.read_csv("data/order_products__train.csv")
order_products_prior_df = pd.read_csv("data/order_products__prior.csv")
orders_df = pd.read_csv("data/orders.csv")
products_df = pd.read_csv("data/products.csv")
aisles_df = pd.read_csv("data/aisles.csv")
departments_df = pd.read_csv("data/departments.csv")
```


```python
orders_df.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>order_id</th>
      <th>user_id</th>
      <th>eval_set</th>
      <th>order_number</th>
      <th>order_dow</th>
      <th>order_hour_of_day</th>
      <th>days_since_prior_order</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2539329</td>
      <td>1</td>
      <td>prior</td>
      <td>1</td>
      <td>2</td>
      <td>8</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2398795</td>
      <td>1</td>
      <td>prior</td>
      <td>2</td>
      <td>3</td>
      <td>7</td>
      <td>15.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>473747</td>
      <td>1</td>
      <td>prior</td>
      <td>3</td>
      <td>3</td>
      <td>12</td>
      <td>21.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2254736</td>
      <td>1</td>
      <td>prior</td>
      <td>4</td>
      <td>4</td>
      <td>7</td>
      <td>29.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>431534</td>
      <td>1</td>
      <td>prior</td>
      <td>5</td>
      <td>4</td>
      <td>15</td>
      <td>28.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
order_products_prior_df.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>order_id</th>
      <th>product_id</th>
      <th>add_to_cart_order</th>
      <th>reordered</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>33120</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>28985</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>9327</td>
      <td>3</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>45918</td>
      <td>4</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2</td>
      <td>30035</td>
      <td>5</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
order_products_train_df.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>order_id</th>
      <th>product_id</th>
      <th>add_to_cart_order</th>
      <th>reordered</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>49302</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>11109</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>10246</td>
      <td>3</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>49683</td>
      <td>4</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>43633</td>
      <td>5</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
products_df.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>product_id</th>
      <th>product_name</th>
      <th>aisle_id</th>
      <th>department_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Chocolate Sandwich Cookies</td>
      <td>61</td>
      <td>19</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>All-Seasons Salt</td>
      <td>104</td>
      <td>13</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Robust Golden Unsweetened Oolong Tea</td>
      <td>94</td>
      <td>7</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Smart Ones Classic Favorites Mini Rigatoni Wit...</td>
      <td>38</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Green Chile Anytime Sauce</td>
      <td>5</td>
      <td>13</td>
    </tr>
  </tbody>
</table>
</div>




```python
aisles_df.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>aisle_id</th>
      <th>aisle</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>prepared soups salads</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>specialty cheeses</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>energy granola bars</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>instant foods</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>marinades meat preparation</td>
    </tr>
  </tbody>
</table>
</div>




```python
departments_df.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>department_id</th>
      <th>department</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>frozen</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>other</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>bakery</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>produce</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>alcohol</td>
    </tr>
  </tbody>
</table>
</div>




```python
cnt_srs = orders_df.eval_set.value_counts()

plt.figure(figsize=(12,8))
sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8)
plt.ylabel('Number of Orders', fontsize=14)
plt.title('Count of rows in each dataset', fontsize=16)
plt.xticks(rotation='horizontal', fontsize=14)
plt.show()
```


![png](output_9_0.png)



```python
def get_unique_count(x):
    return len(np.unique(x))

cnt_srs = orders_df.groupby("eval_set")["user_id"].aggregate(get_unique_count)
cnt_srs
```




    eval_set
    prior    206209
    test      75000
    train    131209
    Name: user_id, dtype: int64




```python
cnt_srs = orders_df.groupby("user_id")["order_number"].aggregate(np.max).reset_index()
cnt_srs = cnt_srs.order_number.value_counts()
cnt_srs = cnt_srs.nlargest(20)

plt.figure(figsize=(12,8))
sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8)
plt.ylabel('Number of Users', fontsize=14)
plt.xlabel('How Many Times Did They Order?', fontsize=14)
plt.xticks(rotation='horizontal')
plt.show()
```


![png](output_11_0.png)



```python
grouped_df = orders_df.groupby(["order_dow", "order_hour_of_day"])["order_number"].aggregate("count").reset_index()
grouped_df = grouped_df.pivot('order_dow', 'order_hour_of_day', 'order_number')

plt.figure(figsize=(16,6))
sns.heatmap(grouped_df,  cmap='Reds')
plt.ylabel('Day of Week', fontsize=14)
plt.xlabel('Hour of Day', fontsize=14)
plt.title("Days of week and hours of day with the most orders", fontsize=16)
plt.show()
```


![png](output_12_0.png)



```python
print 'percentage of reordered products in prior set'
print round(order_products_prior_df.reordered.sum() / float(order_products_prior_df.shape[0]),4)*100, '%'
```

    percentage of reordered products in prior set
    58.97 %



```python
print 'orders with reordered products'
grouped_df = order_products_prior_df.groupby("order_id")["reordered"].aggregate("sum").reset_index()
grouped_df["reordered"].ix[grouped_df["reordered"]>1] = 1
grouped_df.reordered.value_counts() / grouped_df.shape[0]
```

    orders with reordered products





    1    0.879151
    0    0.120849
    Name: reordered, dtype: float64




```python
grouped_df = order_products_train_df.groupby("order_id")["add_to_cart_order"].aggregate("max").reset_index()
cnt_srs = grouped_df.add_to_cart_order.value_counts()
cnt_srs = cnt_srs.nlargest(20)

plt.figure(figsize=(12,8))
sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8)
plt.ylabel('Number of Orders', fontsize=14)
plt.xlabel('What is the Most Frequent Basket Size?', fontsize=14)
plt.xticks(rotation='horizontal')
plt.show()
```


![png](output_15_0.png)



```python
print 'Merge products with prior orders'
order_products_prior_df = pd.merge(order_products_prior_df, products_df, on='product_id')
order_products_prior_df = pd.merge(order_products_prior_df, aisles_df, on='aisle_id')
order_products_prior_df = pd.merge(order_products_prior_df, departments_df, on='department_id')
order_products_prior_df.head()
```

    Merge products with prior orders





<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>order_id</th>
      <th>product_id</th>
      <th>add_to_cart_order</th>
      <th>reordered</th>
      <th>product_name</th>
      <th>aisle_id</th>
      <th>department_id</th>
      <th>aisle</th>
      <th>department</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>33120</td>
      <td>1</td>
      <td>1</td>
      <td>Organic Egg Whites</td>
      <td>86</td>
      <td>16</td>
      <td>eggs</td>
      <td>dairy eggs</td>
    </tr>
    <tr>
      <th>1</th>
      <td>26</td>
      <td>33120</td>
      <td>5</td>
      <td>0</td>
      <td>Organic Egg Whites</td>
      <td>86</td>
      <td>16</td>
      <td>eggs</td>
      <td>dairy eggs</td>
    </tr>
    <tr>
      <th>2</th>
      <td>120</td>
      <td>33120</td>
      <td>13</td>
      <td>0</td>
      <td>Organic Egg Whites</td>
      <td>86</td>
      <td>16</td>
      <td>eggs</td>
      <td>dairy eggs</td>
    </tr>
    <tr>
      <th>3</th>
      <td>327</td>
      <td>33120</td>
      <td>5</td>
      <td>1</td>
      <td>Organic Egg Whites</td>
      <td>86</td>
      <td>16</td>
      <td>eggs</td>
      <td>dairy eggs</td>
    </tr>
    <tr>
      <th>4</th>
      <td>390</td>
      <td>33120</td>
      <td>28</td>
      <td>1</td>
      <td>Organic Egg Whites</td>
      <td>86</td>
      <td>16</td>
      <td>eggs</td>
      <td>dairy eggs</td>
    </tr>
  </tbody>
</table>
</div>




```python
print 'Most frequently bought products'
cnt_srs = order_products_prior_df['product_name'].value_counts().reset_index().head(20)
cnt_srs.columns = ['product_name', 'frequency_count']
cnt_srs
```

    Most frequently bought products





<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>product_name</th>
      <th>frequency_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Banana</td>
      <td>472565</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Bag of Organic Bananas</td>
      <td>379450</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Organic Strawberries</td>
      <td>264683</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Organic Baby Spinach</td>
      <td>241921</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Organic Hass Avocado</td>
      <td>213584</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Organic Avocado</td>
      <td>176815</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Large Lemon</td>
      <td>152657</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Strawberries</td>
      <td>142951</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Limes</td>
      <td>140627</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Organic Whole Milk</td>
      <td>137905</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Organic Raspberries</td>
      <td>137057</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Organic Yellow Onion</td>
      <td>113426</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Organic Garlic</td>
      <td>109778</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Organic Zucchini</td>
      <td>104823</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Organic Blueberries</td>
      <td>100060</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Cucumber Kirby</td>
      <td>97315</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Organic Fuji Apple</td>
      <td>89632</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Organic Lemon</td>
      <td>87746</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Apple Honeycrisp Organic</td>
      <td>85020</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Organic Grape Tomatoes</td>
      <td>84255</td>
    </tr>
  </tbody>
</table>
</div>




```python
grouped_df = order_products_prior_df.groupby(["department"])["reordered"].aggregate("mean").reset_index()

plt.figure(figsize=(12,8))
sns.pointplot(grouped_df['department'].values, grouped_df['reordered'].values, alpha=0.8, color=color[2])
plt.ylabel('Reorder ratio', fontsize=14)
plt.xlabel('Department', fontsize=14)
plt.title("Department - Reorder ratio", fontsize=16)
plt.xticks(rotation='vertical')
plt.show()
```


![png](output_18_0.png)



```python
order_products_prior_df["add_to_cart_order"] = order_products_prior_df["add_to_cart_order"].copy()
order_products_prior_df["add_to_cart_order"].ix[order_products_prior_df["add_to_cart_order"]>70] = 70
grouped_df = order_products_prior_df.groupby(["add_to_cart_order"])["reordered"].aggregate("mean").reset_index()

plt.figure(figsize=(12,8))
sns.pointplot(grouped_df['add_to_cart_order'].values, grouped_df['reordered'].values, alpha=0.8, color=color[2])
plt.ylabel('Reorder ratio', fontsize=14)
plt.xlabel('Add to cart order', fontsize=14)
plt.title("Add to cart order - Reorder ratio", fontsize=16)
plt.xticks(rotation='vertical')
plt.show()
```


![png](output_19_0.png)



```python

```

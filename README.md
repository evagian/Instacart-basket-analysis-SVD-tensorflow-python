Description 

The dataset used for this project purposes consists of 3 million open source online grocery store orders from more than 200 thousands of users. The dataset was available in one of the Kaggle’s competitions named ‘Instacart Market Basket Analysis’. This competition challenged data miners from all over the world to answer to the following question: “Which products will an Instacart consumer purchase in his next basket?”. 

Market basket analysis is an important component of every retail company. Simple, yet powerful - MBA is an inexpensive technique to identify cross-sell opportunities and engage customers. At the same time, personalized recommendation systems differentiate companies from the competition and they can lead to competitive advantages. Moreover, recommendation systems are proven to improve user experience, to increase user traffic and the number of purchases and to encourage user engagement and satisfaction. This competition was an opportunity for us to expand our knowledge and to gain hands on experience on models and techniques used in the fields of basket analysis and recommendation systems.

Data

The dataset used for this project purposes consists of 3 million open source online grocery store orders from more than 200 thousands of users. For each user, it contains between 4 and 100 of their orders, with the sequence of products purchased in each order. It also includes information concerning the week and hour of day that the order was placed, and a relative measure of time between orders. 

The dataset can be found here: https://www.kaggle.com/c/instacart-market-basket-analysis/data. All data files should be placed in the /data directory.

The dataset was too large to be handled and in order to deal with the memory overload problem, we kept the last 6 orders of every user and dropped the rest. We then split the dataset into 2 sets: train set and test set. The test set contained the last order of every user and the train set contained the rest. 

Singular Value Decomposition 

Our aim is to predict the basket size as well as the products bought in the next order of each user. In the first step, we use Singular Value Decomposition (SVD) in order to estimate the size of the basket that we want to predict. Let’s say for example that the estimated basket size equals to n. In the second step we will predict the n products which we believe that the user will buy in his next order. To do so we will use statistic metrics of users’ consumer behavior. We will rank the importance of each product to each user taking under consideration user’s past purchases and preferences. Finally, we will recommend to each user the n products with the highest ranking.

Running the code

Download only the following 2 files from https://www.kaggle.com/c/instacart-market-basket-analysis/data
1. order_products__prior.csv
2. order_products__train.csv
Place them in /data directory

Run python files in the following order:
1. svd_train_val.py
2. print_predictions.py
3. model_evaluation.py


Data Overview 

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
    



```python
order_products_train_df = pd.read_csv("data/order_products__train.csv")
order_products_prior_df = pd.read_csv("data/order_products__prior.csv")
orders_df = pd.read_csv("data/orders.csv")
products_df = pd.read_csv("data/products.csv")
aisles_df = pd.read_csv("data/aisles.csv")
departments_df = pd.read_csv("data/departments.csv")
```
The first file includes a list of all orders, one row per order. For example, we can see that user 1 has 11 orders, 1 of which is in the train set, and 10 of which are prior orders. This file also contains the order number, the day of the week and hour of the day when the order was made and finally the days since the user’s prior order. The orders.csv doesn’t not include which products were purchased in each order. This is contained in the order_products.csv

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


The second and third file specifies which products were purchased in each prior and train order accordingly. Order_products__prior.csv contains previous order contents for all customers, while Order_products__train.csv contains the last order of every user which is the order that we want to predict. The attribute ‘add_to_cart_order’ describes the order in which the user bought one specific product while completing his order. The attribute 'reordered' indicates that the customer has a previous order.that contains the product. 

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

The fourth file contains the names of the products with their corresponding product_id. Furthermore the aisle and department are included.



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

The fifth file contains the different aisles in which the products belong.


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

Finally, the fifth file contains the different departments in which the products belong.


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


The dataset consists of information about 3.4 million grocery orders, distributed across 6 csv files as it was mentioned earlier. There are 206,209 customers in total. Out of which, the last purchase of 131,209 customers are given as train set and we need to predict for the rest 75,000 customers. The products belong to 134 aisles and the aisles belong to 21 departments.


SVD Model Step 1

Singular value decomposition (SVD), Tensorflow and neural networks were used during the first step of predicting the next basket size. SVD is a data dimensionality reduction technique but it can also be used in collaborative filtering. Factorization models such as SVD are very popular in recommendation systems because they can be used to discover latent features underlying the interactions between two different kinds of entities. Tensorflow is a general computation framework using data flow. It provides variant SGD learning algorithms, CPU/GPU acceleration, and distributed training in a computer cluster. Word2vec is a two-layer neural network that processes text. Its input is a text corpus and its output is a set of vectors. Word2vec’s applications automatically learn relationships between two entities and therefore can extend beyond parsing sentences. It can be applied just as well to recommender systems, code, likes, playlists, social media graphs and other verbal or symbolic series in which patterns may be discerned. Since Tensorflow has several embedding modules for word2vc-like application, it is supposed to be a good platform for factorization such as SVD. Finally, neural networks were used to achieve higher accuracy. The model was retrained multiple times, and an error-correction learning rate was applied. 

SVD requires a data matrix A of size nxm as input. In our case, the m rows of matrix A represent the user orders of the train set, the n columns represent the number of ‘days since the previous order’ and the matrix values represent the basket size of the specified order. What SVD does is to represent the matrix A as a product of different matrices U, Σ and V so that A[mxn] = U[mxr]Σ[rxr](V[nxr])T. According to theory, there is always a possible unique way to decompose a real matrix A into three others, UΣVT, where U and V are column orthonormal (sum of the squared values in each column equals 1) and orthogonal (the inner product of their columns equals 0), while Σ is diagonal. Matrix U is called left singular value and has size mxr, Σ is a diagonal matrix of singular values, with size rxr and it has zeros everywhere except from the diagonal. The diagonal contains the positive singular values which are sorted in decreasing order. V stores the right singular vectors with size nxr. In our case, table U describes how possible is that the next basket of user mi is of size ri. Singular values of table Σ represent the strength of every concept, which in our case describe the certainty with which we relate every order with a possible basket size. Matrix V is a concept to ‘day since prior order’ matrix and therefore describes the certainty with which we relate the ‘days since prior order’ of every order with a possible basket size r.

In neural network terminology, batch size is the number of training examples in one forward/backward pass of the training set and one epoch is one forward pass and one backward pass of all the training examples. Furthermore, learning rate is a technique of comparing the system output to the desired output value, and using that error to direct the training. In order to achieve higher accuracy, we split the training set into batches of size 100 and we repeated the training procedure for 100 epochs. A model was trained for every 100 row batch. The whole training process got repeated 100 times. We also defined an error-correction learning rate with value 0.001.

SVD Model Step 2

In this step we will predict the products in the next basket. For every user - product relationship we calculated a rating system based on the user’s past consumer behavior. These statistics include the percentage of baskets in which user x purchased product y, the reorder rate of each product, the average purchase order in which user x purchases product y, the average purchase frequency of each product etc. Every possible user – product relationship was then ranked and sorted in decreasing order. Higher ranks mean that user x will probably buy product y in his next order. Therefore, if in step 1 we had predicted that the size of the next basket of user x equals to 5, then in the second step we would recommend to user x the 5 products with the highest ranking.


Future improvements

Step 2 doesn't know how to cope with missing values. For example, our model will only recommend to users, products that they have bought at least one time in the past. This fact, limits our model accuracy to a ceiling value. As a future improvement of this model we consider also implementing SVD in the second step of the methodology. More specifically we will develop a rating system for every user-product relationship based on the past consumer behaviour. We will then perform SVD to estimate the rating for the products that users have not purchased yet. In this way, when predicting the next basket, we will also be able to recommend to users products that they have not purchased in any time in the past.

Calculation

Software

1.      Python 2.7 – Anaconda
2.      Tensorflow 1.3 – CPU ONLY version

Hardware

1.      Intel i3
2.      12GB DDR3
3.      SSD

Bibliography

Dataset 

Instacart Market Basket Analysis, https://www.kaggle.com/c/instacart-market-basket-analysis/data

SVD model

Guocong Song, TF-recomm. GitHub. Retrieved from https://github.com/songgc/TF-recomm
Yehuda Koren, Factorization Meets the Neighborhood: a Multifaceted Collaborative Filtering Model
Introduction to Word2Vec, https://deeplearning4j.org/word2vec
Recommender systems with TensorFlow, https://theintelligenceofinformation.wordpress.com/2017/05/31/recommender-systems-with-tensorflow/
Singular Value Decomposition | Stanford University, https://www.youtube.com/watch?v=P5mlg91as1c&t=9s
Making sense of word2vec , https://rare-technologies.com/making-sense-of-word2vec/

import time
from collections import deque
import csv
import numpy as np
import tensorflow as tf
from six import next
from tensorflow.core.framework import summary_pb2
import pandas as pd
import dataio
import ops

# This model is trained to predict the size of the next basket of every user
# input: "user_id", "days_since_prior_order", "basket_size", "order_id"
# output: the basket size of the user's next order

np.random.seed(13575)

BATCH_SIZE = 100
USER_NUM =  206202
ITEM_NUM = 32
DIM = 15
EPOCH_MAX = 100
DEVICE = "/cpu:0"


def clip(x):
    return np.clip(x, 1.0, None)


def make_scalar_summary(name, val):
    return summary_pb2.Summary(value=[summary_pb2.Summary.Value(tag=name, simple_value=val)])


def get_data():
    df = dataio.read_process("data/user_basket_size.csv", sep=",")
    df['group_indicator'] = (df.ix[:,0] != df.ix[:,0].shift(-1)).astype(int)

    df_train = df.loc[df.group_indicator==0]
    df_train = df_train.drop('group_indicator', axis=1)

    df_test =  df.loc[df.group_indicator==1]
    df_test = df_test.drop('group_indicator', axis=1)
    df = df.drop('group_indicator', axis=1)

    return df_train, df_test


def svd(train, test):
    samples_per_batch = len(train) // BATCH_SIZE
    print test.head(10)
    iter_train = dataio.ShuffleIterator([train["user"],
                                         train["days_since_prior_order"],
                                         train["basket_size"]],
                                        batch_size=BATCH_SIZE)

    iter_test = dataio.OneEpochIterator([test["user"],
                                         test["days_since_prior_order"],
                                         test["basket_size"]],
                                        batch_size=-1)

    user_batch = tf.placeholder(tf.int32, shape=[None], name="id_user")
    days_since_prior_order_batch = tf.placeholder(tf.int32, shape=[None], name="id_days_since_prior_order")
    basket_size_batch = tf.placeholder(tf.float32, shape=[None])

    infer, regularizer = ops.inference_svd(user_batch, days_since_prior_order_batch, user_num=USER_NUM, item_num=ITEM_NUM, dim=DIM,
                                           device=DEVICE)
    global_step = tf.contrib.framework.get_or_create_global_step()
    _, train_op = ops.optimization(infer, regularizer, basket_size_batch, learning_rate=0.001, reg=0.05, device=DEVICE)

    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_op)
        summary_writer = tf.summary.FileWriter(logdir="/tmp/svd/log", graph=sess.graph)
        print("{} {} {} {}".format("epoch", "train_error", "val_error", "elapsed_time"))
        errors = deque(maxlen=samples_per_batch)
        start = time.time()
        min = 100
        predList = []
        actList = []
        finalPred = []
        finalAct = []
        finalpr = []
        finalac = []
        for i in range(EPOCH_MAX * samples_per_batch):

            users, days_since_prior_orders, basket_sizes = next(iter_train)
            _, pred_batch = sess.run([train_op, infer], feed_dict={user_batch: users,
                                                                   days_since_prior_order_batch: days_since_prior_orders,
                                                                   basket_size_batch: basket_sizes})
            pred_batch = clip(pred_batch)
            errors.append(np.power(pred_batch - basket_sizes, 2))
            if i % samples_per_batch == 0:
                train_err = np.sqrt(np.mean(errors))
                test_err2 = np.array([])
                for users, days_since_prior_orders, basket_sizes in iter_test:
                    pred_batch = sess.run(infer, feed_dict={user_batch: users,
                                                            days_since_prior_order_batch: days_since_prior_orders})
                    #pred_batch = clip(pred_batch)
                    test_err2 = np.append(test_err2, np.power(pred_batch - basket_sizes, 2))

                    pr = pred_batch
                    ac = basket_sizes
                end = time.time()
                test_err = np.sqrt(np.mean(test_err2))
                print("{:3d} {:f} {:f} {:f}(s)".format(i // samples_per_batch, train_err, test_err,
                                                       end - start))
                train_err_summary = make_scalar_summary("training_error", train_err)
                test_err_summary = make_scalar_summary("test_error", test_err)
                summary_writer.add_summary(train_err_summary, i)
                summary_writer.add_summary(test_err_summary, i)
                start = end

                if train_err < min:
                    min = train_err
                    finalpr = pr
                    finalac = ac

        return finalpr, finalac


if __name__ == '__main__':
    df_train, df_test = get_data()
    pr, ac = svd(df_train, df_test)
    print pr, type(pr), ac, type(ac)
    prdf = pd.DataFrame(pr)
    acdf = pd.DataFrame(ac)
    print df_test.head(10)

    result = pd.concat([prdf, acdf], axis=1)

    #filename = 'data/pred-actual.csv'
    #result.to_csv(filename, index=False, encoding='utf-8', header=False)




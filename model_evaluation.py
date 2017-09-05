# input: 2 list of lists
# first list contains the actual products
# second list contains the predicted products

# output: model evaluation metrics
# true positives
# false positives
# false negatives
# total products bought
# accuracy
# precision
# recall
# f1

import csv

#preds = pd.read_csv("pred_products2.csv")
predss = []


# read CSV file & load into list
with open("data/pred_products.csv", 'r') as my_file:
    reader = csv.reader(my_file, delimiter='\t')
    preds = list(reader)

with open("data/actual_products.csv", 'r') as my_file:
    reader = csv.reader(my_file, delimiter='\t')
    acts = list(reader)

acts = [l[0] for l in acts]

TTP = 0
TFP = 0
TFN = 0
TT = 0

i= 0


for pred, act in zip(preds, acts):
        act = str(act)
        act = act.replace(" ", "")
        pred = str(pred)

        pred = pred.replace(" ", "")
        pred = pred.replace("'", "")
        pred = pred.replace("[", "")
        pred = pred.replace("]", "")
        act = act.replace("[", "")
        act = act.replace("]", "")

        act = act.split(",")
        pred = pred.split(",")


        pred = set(pred)
        act = set(act)

        TP = len(set.intersection(act, pred))

        UN = len(set.union(act, pred))

        FP = len(pred)-TP
        FN = len(act)-TP
        T = len(act)

        AC = TP/float(T)
        #print TP, UN, FP, FN, T
        TTP=TTP+TP
        TFP=TFP+FP
        TFN=TFN+FN
        TT=TT+T
        #print TTP, TFP, TFN, TT
TAC = TTP/float(TT)
#print TTP,TT
PRE = TTP/float((TTP+TFP))
REC = TTP/float((TTP+TFN))
F1 = (2*(PRE*REC))/float((PRE+REC))

i = i+1
print 'true positives', TTP, '\nfalse positives', TFP, '\nfalse negatives', TFN, '\ntotal products bought', TT
print '\naccuracy', TAC, '\nprecision', PRE, '\nrecall', REC, '\nf1', F1

'''
true positives 3673 
false positives 6305 
false negatives 6909 
total products bought 10582

accuracy 0.347098847099 
precision 0.368109841652 
recall 0.347098847099 
f1 0.357295719844
'''
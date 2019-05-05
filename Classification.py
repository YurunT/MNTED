from sklearn.svm import SVC
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from imblearn.combine import SMOTEENN
import math
import numpy as np
import scipy.io as sio
import random
import matplotlib.pyplot as plt

file_name_dic = {'aminer': 'aminer/aminer_duicheng',
                 "blog_perturb": 'blog_perturb/blog_perturb',
                 "congress": 'congress/congress_',
                 "Disney": 'Disney/disney_disturb_',
                 "flickr": 'flickr/flickr_disturb_',
                 "wiki": 'wiki/wiki_disturb_'}

# dataset_name = "aminer"
# dataset_name = "blog_perturb"
# dataset_name = "congress"
# dataset_name = "Disney"
dataset_name = "flickr"
# dataset_name = "wiki"

iterateTime = 5
folds = 10


def load_data(file_name):
    print("Dataset:", file_name)
    print("Embedding Loading.....")
    embedding_contents = sio.loadmat("result/"+file_name+str(0)+"_"+"Embedding.mat")
    mat_contents = sio.loadmat(file_name+'1'+'.mat')
    i = 0
    j=1
    Label = list()
    G_MNTED = list()
    G_NET = list()
    while embedding_contents is not None:
        i = i+1
        j = j+1
        V_MNTED = embedding_contents['V_MNTED']
        V_NET = embedding_contents['V_Net']
        if file_name == 'Disney/disney_disturb_':
            label = mat_contents["best_gnd"]
        else:
            label = mat_contents["gnd"]

        # if 'float' not in str(network.dtype) or 'double' not in str(network.dtype):
        #     network = network.astype(float)
        G_MNTED.append(V_MNTED)
        G_NET.append(V_NET)
        list_label=list()
        for l in label:
            list_label.append(l[0])
        Label.append(list_label)
        del embedding_contents
        del mat_contents
        try:
            embedding_contents = sio.loadmat("result/"+file_name+str(i)+'_Embedding.mat')
            mat_contents = sio.loadmat(file_name + str(j) + '.mat')
        except FileNotFoundError:
            break
    print("Data loaded!")
    return G_MNTED, G_NET, Label


def twoLabel_classification(train_x, test_x, train_label, test_label):
    print("doing svm...")
    clf = SVC(kernel='poly', C=0.4)
    clf.fit(train_x, train_label)

    # pred_label = clf.predict(test_x)
    pred_label = clf.predict(train_x)
    print(train_label)
    print(pred_label)
    print("1 in pred_label :",1 in pred_label)
    prec_score=precision_score(train_label,pred_label,average='micro')
    recall=recall_score(train_label,pred_label,pos_label=1)
    f_measure=f1_score(train_label,pred_label,pos_label=1)
    auc_score=roc_auc_score(train_label,pred_label,average='micro')
    fpr, tpr,thresholds=roc_curve(train_label,pred_label,pos_label=1)
    auc_score = auc(fpr, tpr)
    plt.plot(fpr,tpr,marker='o')
    plt.show()
    acc_score = accuracy_score(train_label,pred_label)
    print("acc:",acc_score)
    print("precision:",prec_score)
    print("recall:",recall)
    print("f-1:",f_measure)
    print("auc:",auc_score)
    res_array = np.array([acc_score,prec_score,recall,f_measure,auc_score])
    return res_array


def generate_10_fold_data(data_length):
    split_interval = int(data_length/folds)
    full_index = list(np.arange(data_length))
    Groups = list()
    for i in range(folds-1):
        Groups.append(random.sample(full_index, split_interval))
        full_index = list(set(full_index).difference(set(Groups[i])))
    Groups.append(full_index)
    return Groups


# def upsampling(G_MNTED,G_NET,Label,tfrate):
#     time_len=len(G_MNTED)
#     new_G_MNTED=list()
#     new_G_NET=list()
#     new_Label=list()
#     for day in range(time_len):
#         day_gmnted=G_MNTED[day]
#         day_gnet=G_NET[day]
#         day_label=Label[day]
#
#         day_label_arr=np.array(day_label)
#         indices_negative=np.where(day_label_arr==1)[0].tolist()
#         indices=np.arange(len(day_label)).tolist()
#
#         indices_negative_count=len(indices_negative)
#         indices_positive_count=len(day_label_arr)-indices_negative_count
#
#         upsamplingVolume=int(indices_positive_count/tfrate)-indices_negative_count# make positive/negative approximate 2
#         if upsamplingVolume>=indices_negative_count:
#             times=math.floor(upsamplingVolume/indices_negative_count)
#             [indices.extend(indices_negative) for __ in range(times)]
#             left=upsamplingVolume-indices_negative_count*times
#             if left>0:
#                 indices.extend(indices_negative[:left])
#         else:
#             indices.extend(indices_negative[:upsamplingVolume])
#
#         new_day_label=list()
#         new_day_gmnted=list()
#         new_day_gnet=list()
#
#         for i in indices:
#             new_day_label.append(day_label[i])
#             new_day_gmnted.append(day_gmnted[i])
#             new_day_gnet.append(day_gnet[i])
#         new_Label.append(new_day_label)
#         new_G_MNTED.append(new_day_gmnted)
#         new_G_NET.append(new_day_gnet)
#     return new_G_MNTED,new_G_NET,new_Label


file_name = file_name_dic[dataset_name]
G_MNTED_origin, G_NET_origin, Label_origin = load_data(file_name)  # origin embeddings without upsampling

# upsample embeddings using SMOTEENN
sm = SMOTEENN()
methodList = {"G_MNTED": G_MNTED_origin, "G_NET": G_NET_origin}
G_MNTED = list()
G_NET = list()
Label_MNTED = list()
Label_NET = list()
print("Start upsampling using SMOTEENN")
for method, em in methodList.items():
    for i in range(len(em)):
        g_new, label_new = sm.fit_sample(em[i], Label_origin[i])
        if method == "G_MNTED":
            G_MNTED.append(g_new)
            Label_MNTED.append(label_new)
        else:
            G_NET.append(g_new)
            Label_NET.append(label_new)
print("Finish unsampling!")

days = len(G_MNTED)
data=open("result/"+file_name+"_"+"classification_result.txt", 'a+')
print("Dataset:", dataset_name, file=data)
print("Total time length:", days, file=data)
print("%d x %d fold cross-validation used for each day\n" % (iterateTime, folds), file=data)
data.close()

days_copy = days

total_days_metric_array_MNTED = np.zeros(5)
total_days_metric_array_NET = np.zeros(5)

for day in range(days_copy):
    print("This is day", day)
    day_avg_metric_array_MNTED = np.zeros(5)
    day_avg_metric_array_NET = np.zeros(5)
    for method, em in methodList.items():
        print("This is method :", method)
        if method == "G_MNTED":
            label_method = Label_MNTED
        else:
            label_method = Label_NET

        Label_day_arr = np.array(label_method[day])
        indices_1 = np.where(Label_day_arr == 1)[0]
        if len(indices_1) < 2:
            raise ValueError("negative sample less than 2!")

        metric_sum_array = np.zeros(5)  # refer to precision_sum,recall_sum,f1_sum,auc_sum respectively
        V=em[day]

        for ite in range(iterateTime):
            print("This is iterate time:", ite)
            # class_num=len(Labels)
            train_x = list()
            test_x = list()
            test_index = list()
            train_index=list()
            train_label = list()
            test_label = list()

            Groups = generate_10_fold_data(V.shape[0])
            for i in range(folds):
                test_index = []
                train_index = []
                train_label = []
                test_label = []
                test_index = Groups[i]
                [train_index.extend(Groups[j]) for j in range(len(Groups)) if j != i]
                [train_label.append(label_method[day][k]) for k in train_index]
                [test_label.append(label_method[day][k]) for k in test_index]

                sum_train = sum(np.array(train_label))
                sum_test = sum(np.array(test_label))
                if sum_train == 0:
                    index_1=test_label.index(1)
                    train_index.append(test_index[index_1])
                    test_index.pop(index_1)
                if sum_test == 0:
                    index_1=train_label.index(1)
                    test_index.append(train_index[index_1])
                    train_index.pop(index_1)
                train_label = []
                test_label = []
                [train_label.append(label_method[day][k]) for k in train_index]
                [test_label.append(label_method[day][k]) for k in test_index]

                res_array = twoLabel_classification(V[train_index, ],
                                                    V[test_index, ],
                                                    train_label,
                                                    test_label)
                metric_sum_array += res_array

                test_index = []
                train_index = []
                train_label = []
                test_label = []
        if method == "G_MNTED":
            day_avg_metric_array_MNTED += metric_sum_array/(iterateTime*folds)
        else:
            day_avg_metric_array_NET += metric_sum_array/(iterateTime*folds)

    data=open("result/"+file_name+"_"+"classification_result.txt", 'a+')
    print("DAY %d:" % (day+1), file=data)
    print("\t\tavg acc-avg precision-avg recall-avg f1-avg auc", file=data)
    print("MNTED:\t%.12f\t%.12f\t%.12f\t%.12f\t%.12f" % (day_avg_metric_array_MNTED[0],
                                                         day_avg_metric_array_MNTED[1],
                                                         day_avg_metric_array_MNTED[2],
                                                         day_avg_metric_array_MNTED[3],
                                                         day_avg_metric_array_MNTED[4]), file=data)
    print("N E T:\t%.12f\t%.12f\t%.12f\t%.12f\t%.12f" % (day_avg_metric_array_MNTED[0],
                                                         day_avg_metric_array_MNTED[1],
                                                         day_avg_metric_array_MNTED[2],
                                                         day_avg_metric_array_MNTED[3],
                                                         day_avg_metric_array_MNTED[4]), file=data)

    data.close()

    total_days_metric_array_MNTED += day_avg_metric_array_MNTED
    total_days_metric_array_NET += day_avg_metric_array_NET


total_days_avg_score_MNTED = total_days_metric_array_MNTED/days
total_days_avg_score_NET = total_days_metric_array_NET/days
data=open("result/"+file_name+"_"+"classification_result.txt", 'a+')
print("TOTAL:", file=data)
print("\t\tavg acc-avg precision-avg recall-avg f1-avg auc", file=data)
print("MNTED:\t%.12f\t%.12f\t%.12f\t%.12f\t%.12f" % (total_days_avg_score_MNTED[0],
                                                     total_days_avg_score_MNTED[1],
                                                     total_days_avg_score_MNTED[2],
                                                     total_days_avg_score_MNTED[3],
                                                     total_days_avg_score_MNTED[4]), file=data)
print("N E T:\t%.12f\t%.12f\t%.12f\t%.12f\t%.12f" % (total_days_avg_score_NET[0],
                                                     total_days_avg_score_NET[1],
                                                     total_days_avg_score_NET[2],
                                                     total_days_avg_score_NET[3],
                                                     total_days_avg_score_NET[4]), file=data)

data.close()
print("classification done!")




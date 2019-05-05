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

iterateTime=5
folds=10


def load_data(file_name):
    print("Dataset:", file_name)
    print("Embedding Loading.....")
    embedding_contents = sio.loadmat("result/"+file_name+str(0)+"_"+"Embedding.mat")
    mat_contents = sio.loadmat(file_name+'1'+'.mat')
    i = 0
    j=1
    Label = list()
    G_MNTED=list()
    G_NET=list()
    while embedding_contents is not None:
        i = i+1
        j=j+1
        V_MNTED = embedding_contents['V_MNTED']
        V_NET = embedding_contents['V_Net']
        if file_name=='Disney/disney_disturb_':
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
        # Label = embedding_contents["Label"]
        del embedding_contents
        del mat_contents
        try:
            embedding_contents = sio.loadmat("result/"+file_name+str(i)+'_Embedding.mat')
            mat_contents = sio.loadmat(file_name + str(j) + '.mat')
        except FileNotFoundError:
            break
    print("Data loaded!")
    return G_MNTED, G_NET, Label


def twoLabel_classification(train_x, test_x,train_label,test_label):
    print("doing svm...")
    clf = SVC(kernel='poly', C=0.4)
    clf.fit(train_x, train_label)
    pred_label = clf.predict(test_x)
    print("pred_label",pred_label)
    # log = MultiOutputClassifier(SGDClassifier(loss="log"), n_jobs=10)
    # log.fit(train_x, train_y)
    # pred_y=log.predict(test_x)
    print("test_label:",test_label)
    print("test_label==pred_label:",test_label==pred_label)
    print("1 in pred_label :",1 in pred_label)
    prec_score=precision_score(test_label,pred_label,average='micro' )
    recall=recall_score(test_label,pred_label,pos_label=0)
    f_measure=f1_score(test_label,pred_label,pos_label=0)

    fpr, tpr,thresholds=roc_curve(test_label,pred_label,pos_label=0)
    auc_score = auc(fpr, tpr)
    plt.plot(fpr,tpr,marker='o')
    plt.show()
    acc = accuracy_score(test_label,pred_label)
    print("acc:",acc)
    print("precision:",prec_score)
    print("recall:",recall)
    print("f-1:",f_measure)
    print("auc:",auc_score)

    return prec_score,recall,f_measure,auc_score


def generate_10_fold_data(data_length):
    split_interval=int(data_length/folds)
    full_index=list(np.arange(data_length))
    Groups=list()
    for i in range(folds-1):
        Groups.append(random.sample(full_index,split_interval))
        full_index = list(set(full_index).difference(set(Groups[i])))
    Groups.append(full_index)
    return Groups


def upsampling(G_MNTED,G_NET,Label,tfrate):
    time_len=len(G_MNTED)
    new_G_MNTED=list()
    new_G_NET=list()
    new_Label=list()
    for day in range(time_len):
        day_gmnted=G_MNTED[day]
        day_gnet=G_NET[day]
        day_label=Label[day]

        day_label_arr=np.array(day_label)
        indices_negative=np.where(day_label_arr==1)[0].tolist()
        indices=np.arange(len(day_label)).tolist()

        indices_negative_count=len(indices_negative)
        indices_positive_count=len(day_label_arr)-indices_negative_count

        upsamplingVolume=int(indices_positive_count/tfrate)-indices_negative_count# make positive/negative approximate 2
        if upsamplingVolume>=indices_negative_count:
            times=math.floor(upsamplingVolume/indices_negative_count)
            [indices.extend(indices_negative) for __ in range(times)]
            left=upsamplingVolume-indices_negative_count*times
            if left>0:
                indices.extend(indices_negative[:left])
        else:
            indices.extend(indices_negative[:upsamplingVolume])

        new_day_label=list()
        new_day_gmnted=list()
        new_day_gnet=list()

        for i in indices:
            new_day_label.append(day_label[i])
            new_day_gmnted.append(day_gmnted[i])
            new_day_gnet.append(day_gnet[i])
        new_Label.append(new_day_label)
        new_G_MNTED.append(new_day_gmnted)
        new_G_NET.append(new_day_gnet)
    return new_G_MNTED,new_G_NET,new_Label









file_name=file_name_dic[dataset_name]
G_MNTED,G_NET,Label=load_data(file_name)
# G_MNTED_origin,G_NET_origin,Label_origin=load_data(file_name)  # origin embeddings without upsampling

# upsample embeddings using SMOTEENN
# sm=SMOTEENN()
# methodList={"G_MNTED":G_MNTED_origin,"G_NET":G_NET_origin}
# G_MNTED=list()
# G_NET=list()
# Label_MNTED=list()
# Label_NET=list()
# for method,em in methodList:
#     for i in len(em):
#         g_new,label_new=sm.fit_sample(em[i],Label_origin[i])
#         if method=="G_MNTED":
#             G_MNTED.append(g_new)
#             Label_MNTED.append(label_new)
#         else:
#             G_NET.append(g_new)
#             Label_NET.append(label_new)




days=len(G_MNTED)

total_days_precision_score_MNTED=0
total_days_precision_score_NET=0
total_days_recall_score_MNTED=0
total_days_recall_score_NET=0
total_days_f1_score_MNTED=0
total_days_f1_score_NET=0
total_days_auc_score_MNTED=0
total_days_auc_score_NET=0

data=open("result/"+file_name+"_"+"classification_result.txt",'a+')
print("Dataset:",dataset_name,file=data)
print("Total time length:",days,file=data)
print("%d x %d fold cross-validation used for each day\n"  %(iterateTime , folds),file=data)
data.close()
days_copy=days
for day in range(days_copy):
    print("This is day", day)
    #num of categories
    # Labels=[]
    # for label in Label[day]:
    #     if label not in Labels:
    #         Labels.append(label)
    Label_day_arr_MNTED = np.array(Label[day])
    indices_1 = np.where(Label_day_arr_MNTED == 1)[0]
    if len(indices_1) < 2:
        # raise ValueError("negative sample less than 2!")
        print("skip day %d , for negative sample less than 2! ",day)
        days-=1
        continue
    precision_sum_MNTED = 0
    precision_sum_NET = 0
    recall_sum_MNTED=0
    recall_sum_NET=0
    f1_sum_MNTED=0
    f1_sum_NET=0
    auc_sum_MNTED=0
    auc_sum_NET=0
    # V_MNTED=np.reshape(G_MNTED[day],(len(G_MNTED[day]),G_MNTED[day][0].shape[0]))#G_MNTED[day]
    # V_NET=np.reshape(G_NET[day],(len(G_NET[day]),G_NET[day][0].shape[0]))
    V_MNTED=G_MNTED[day]
    V_NET=G_NET[day]

    for ite in range(iterateTime):
        print("This is iterate time:",ite)
        # class_num=len(Labels)
        train_x = list()
        test_x = list()
        test_index = list()
        train_index=list()
        train_label = list()
        test_label = list()

        Groups = generate_10_fold_data(V_MNTED.shape[0])
        for i in range(folds):
            test_index = []
            train_index = []
            train_label = []
            test_label = []
            test_index = Groups[i]
            [train_index.extend(Groups[j]) for j in range(len(Groups)) if j != i]
            [train_label.append(Label[day][k]) for k in train_index]
            [test_label.append(Label[day][k]) for k in test_index]

            sum_train=sum(np.array(train_label))
            sum_test=sum(np.array(test_label))
            if sum_train==0:
                index_1=test_label.index(1)
                train_index.append(test_index[index_1])
                test_index.pop(index_1)
            if sum_test==0:
                index_1=train_label.index(1)
                test_index.append(train_index[index_1])
                train_index.pop(index_1)
            train_label=[]
            test_label=[]
            [train_label.append(Label[day][k]) for k in train_index]
            [test_label.append(Label[day][k]) for k in test_index]

            precision_V_MNTED, recall_V_MNTED,f1_V_MNTED,auc_V_MNTED=twoLabel_classification(V_MNTED[train_index,],
                                                                                             V_MNTED[test_index,],
                                                                                             train_label,
                                                                                             test_label)
            precision_V_NET, recall_V_NET, f1_V_NET, auc_V_NET = twoLabel_classification(V_NET[train_index,],
                                                                                                 V_NET[test_index,],
                                                                                                 train_label,
                                                                                                 test_label)

            precision_sum_MNTED += precision_V_MNTED
            precision_sum_NET += precision_V_NET

            recall_sum_MNTED+=recall_V_MNTED
            recall_sum_NET+=recall_V_NET

            f1_sum_MNTED+=f1_V_MNTED
            f1_sum_NET+=f1_V_NET

            auc_sum_MNTED+=auc_V_MNTED
            auc_sum_NET+=auc_V_NET

            test_index = []
            train_index = []
            train_label = []
            test_label = []
    day_avg_precision_score_MNTED=precision_sum_MNTED/(iterateTime*folds)
    day_avg_precision_score_NET=precision_sum_NET/(iterateTime*folds)
    total_days_precision_score_MNTED+=day_avg_precision_score_MNTED
    total_days_precision_score_NET+=day_avg_precision_score_NET

    day_avg_recall_score_MNTED=recall_sum_MNTED/(iterateTime*folds)
    day_avg_recall_score_NET=recall_sum_NET/(iterateTime*folds)
    total_days_recall_score_MNTED+=day_avg_recall_score_MNTED
    total_days_recall_score_NET+=day_avg_recall_score_NET

    day_avg_f1_score_MNTED=precision_sum_MNTED/(iterateTime*folds)
    day_avg_f1_score_NET=precision_sum_NET/(iterateTime*folds)
    total_days_f1_score_MNTED+=day_avg_f1_score_MNTED
    total_days_f1_score_NET+=day_avg_f1_score_NET

    day_avg_auc_score_MNTED=auc_sum_MNTED/(iterateTime*folds)
    day_avg_auc_score_NET=auc_sum_NET/(iterateTime*folds)
    total_days_auc_score_MNTED+=day_avg_auc_score_MNTED
    total_days_auc_score_NET+=day_avg_auc_score_NET

    data=open("result/"+file_name+"_"+"classification_result.txt",'a+')
    print("DAY %d:" %(day+1),file=data)
    print('MNTED: average precision score : %.12f ' %(day_avg_precision_score_MNTED),file=data)
    print('NET: average precision score : %.12f ' % (day_avg_precision_score_NET), file=data)

    print('MNTED: average recall score : %.12f ' %(day_avg_recall_score_MNTED),file=data)
    print('NET: average recall score : %.12f ' % (day_avg_recall_score_NET), file=data)

    print('MNTED: average f-1 score : %.12f ' %(day_avg_f1_score_MNTED),file=data)
    print('NET: average f-1 score : %.12f ' % (day_avg_f1_score_NET), file=data)

    print('MNTED: average auc score : %.12f ' %(day_avg_auc_score_MNTED),file=data)
    print('NET: average auc score : %.12f \n' % (day_avg_auc_score_NET), file=data)

    data.close()

total_days_avg_score_MNTED=total_days_precision_score_MNTED/days
total_days_avg_score_NET = total_days_precision_score_NET/days
data=open("result/"+file_name+"_"+"classification_result.txt",'a+')
print("TOTAL:",file=data)
print('MNTED: average precision score for the whole time: %.12f ' %(total_days_precision_score_MNTED/days),file=data)
print('NET: average precision score for the whole time: %.12f ' %(total_days_precision_score_NET/days),file=data)

print('MNTED: average recall score for the whole time: %.12f ' %(total_days_recall_score_MNTED/days),file=data)
print('NET: average recall score for the whole time: %.12f ' %(total_days_recall_score_NET/days),file=data)

print('MNTED: average f-1 score for the whole time: %.12f ' %(total_days_f1_score_MNTED/days),file=data)
print('NET: average f-1 score for the whole time: %.12f ' %(total_days_f1_score_NET/days),file=data)

print('MNTED: average f-1 score for the whole time: %.12f ' %(total_days_f1_score_MNTED/days),file=data)
print('NET: average f-1 score for the whole time: %.12f ' %(total_days_f1_score_NET/days),file=data)

print('MNTED: average auc score for the whole time: %.12f ' %(total_days_auc_score_MNTED/days),file=data)
print('NET: average auc score for the whole time: %.12f ' %(total_days_auc_score_NET/days),file=data)

data.close()
print("classification done!")




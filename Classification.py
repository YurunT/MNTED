from sklearn.svm import SVC
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from imblearn.combine import SMOTEENN
from sklearn.model_selection import StratifiedKFold
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


def twoLabel_classification(kerneli,train_x, test_x, train_label, test_label):
    print("doing svm...")
    clf = SVC(kernel=kerneli, C=0.4)
    clf.fit(train_x, train_label)
    pred_label = clf.predict(test_x)

    # pred_label = clf.predict(train_x)
    print(train_label)
    print(pred_label)
    print("1 in pred_label :",1 in pred_label)

    print("0 in ored_label :",0 in pred_label)
    prec_score=precision_score(test_label,pred_label,average='micro')
    recall=recall_score(test_label,pred_label,pos_label=1)
    f_measure=f1_score(test_label,pred_label,pos_label=1)
    # auc_score=roc_auc_score(test_label,pred_label,average='micro')
    fpr, tpr, thresholds = roc_curve(test_label, pred_label,pos_label=1)
    auc_score = auc(fpr, tpr)
    fig=plt.figure(num=1,figsize=(8,6))
    plt.plot(fpr, tpr, marker='o')
    plt.show()

    plt.close()
    acc_score = accuracy_score(test_label,pred_label)
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


file_name = file_name_dic[dataset_name]
G_MNTED_origin, G_NET_origin, Label_origin = load_data(file_name)  # origin embeddings without upsampling

# upsample embeddings using SMOTEENN
sm = SMOTEENN()

methodList = { "G_NET": G_NET_origin,"G_MNTED": G_MNTED_origin}
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
new_methodList={ "G_NET": G_NET,"G_MNTED": G_MNTED}

days = len(G_NET)

data=open("result/"+file_name+"_"+"classification_result.txt", 'w+')
print("Dataset:", dataset_name, file=data)
print("Total time length:", days, file=data)
print("%d x %d fold cross-validation used for each day\n" % (iterateTime, folds), file=data)
data.close()

days_copy = days

total_days_metric_array_MNTED = np.zeros(5)
total_days_metric_array_NET = np.zeros(5)

kernellist=['linear', 'rbf', 'sigmoid']
for kerneli in kernellist:
    print("*******************************This is kernel", kerneli, "*************************************************")
    for day in range(days_copy):
        print("***********************************This is day", day,"*********************************************")
        day_avg_metric_array_MNTED = np.zeros(5)
        day_avg_metric_array_NET = np.zeros(5)
        for method, em in new_methodList.items():
            #print("This is method :", method)
            if method == "G_MNTED":
                label_method = Label_MNTED
            else:
                label_method = Label_NET

            label=label_method[day]
            Label_day_arr = np.array(label)
            indices_1 = np.where(Label_day_arr == 1)[0]
            if len(indices_1) < 2:
                raise ValueError("negative sample less than 2!")

            metric_sum_array = np.zeros(5)  # refer to precision_sum,recall_sum,f1_sum,auc_sum respectively
            V=em[day]

            for ite in range(iterateTime):
                print("******************************This is iterate time:", ite, "******************************")
                skf = StratifiedKFold(n_splits=folds)
                indices = skf.split(V, label)
                for train_index, test_index in indices:
                    print("TRAIN:", train_index, "TEST:", test_index)
                    train_x, test_x = V[train_index], V[test_index]
                    train_label, test_label = label[train_index], label[test_index]
                    res_array = twoLabel_classification(kerneli,
                                                        train_x,
                                                        test_x,
                                                        train_label,
                                                        test_label)
                    metric_sum_array += res_array

            if method == "G_MNTED":
                day_avg_metric_array_MNTED += metric_sum_array / (iterateTime * folds)
            else:
                day_avg_metric_array_NET += metric_sum_array / (iterateTime * folds)

        data = open("result/" + file_name + "_" + "classification_result.txt", 'a+')
        print("DAY %d:" % (day + 1), file=data)
        print("\t\tavg             acc-avg         precision-avg   recall-avg     f1-avg auc", file=data)
        print("MNTED:\t%.12f\t%.12f\t%.12f\t%.12f\t%.12f" % (day_avg_metric_array_MNTED[0],
                                                             day_avg_metric_array_MNTED[1],
                                                             day_avg_metric_array_MNTED[2],
                                                             day_avg_metric_array_MNTED[3],
                                                             day_avg_metric_array_MNTED[4]), file=data)
        print("N E T:\t%.12f\t%.12f\t%.12f\t%.12f\t%.12f" % (day_avg_metric_array_NET[0],
                                                             day_avg_metric_array_NET[1],
                                                             day_avg_metric_array_NET[2],
                                                             day_avg_metric_array_NET[3],
                                                             day_avg_metric_array_NET[4]), "\n", file=data)

        data.close()

        total_days_metric_array_MNTED += day_avg_metric_array_MNTED
        total_days_metric_array_NET += day_avg_metric_array_NET


    total_days_avg_score_MNTED = total_days_metric_array_MNTED/days
    total_days_avg_score_NET = total_days_metric_array_NET/days
    data=open("result/"+file_name+"_"+"classification_result.txt", 'a+')
    print("Kernel:",kerneli, file=data)
    print("TOTAL:", file=data)
    print("\t\tavg             acc-avg         precision-avg   recall-avg     f1-avg auc", file=data)
    print("MNTED:\t%.12f\t%.12f\t%.12f\t%.12f\t%.12f" % (total_days_avg_score_MNTED[0],
                                                         total_days_avg_score_MNTED[1],
                                                         total_days_avg_score_MNTED[2],
                                                         total_days_avg_score_MNTED[3],
                                                         total_days_avg_score_MNTED[4]), file=data)
    print("N E T:\t%.12f\t%.12f\t%.12f\t%.12f\t%.12f" % (total_days_avg_score_NET[0],
                                                         total_days_avg_score_NET[1],
                                                         total_days_avg_score_NET[2],
                                                         total_days_avg_score_NET[3],
                                                         total_days_avg_score_NET[4]), "\n", file=data)



    data.close()
print("classification done!")




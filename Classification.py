from sklearn.svm import SVC
# from sklearn.metrics import classification_report
from sklearn.metrics import precision_score
import numpy as np
import scipy.io as sio
import random

file_name_dic = {'aminer': 'aminer/aminer_duicheng',
                 "blog_perturb": 'blog_perturb/blog_perturb',
                 "congress": 'congress/congress_',
                 "Disney": 'Disney/disney_disturb_',
                 "flickr": 'flickr/flickr_disturb_',
                 "wiki": 'wiki/wiki_disturb_'}
# dataset_name = "aminer"
# dataset_name = "blog_perturb"
# dataset_name = "congress"
dataset_name = "Disney"
# dataset_name = "flickr"
# dataset_name = "wiki"
iterateTime=5
folds=10
def load_data(file_name):
    print("Dataset:", file_name)
    print("Embedding Loading.....")
    embedding_contents = sio.loadmat(file_name+'_Embedding'+str(0)+'.mat')
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
        label = mat_contents["best_gnd"]
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
            embedding_contents = sio.loadmat(file_name+'_Embedding'+str(i)+'.mat')
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
    # log = MultiOutputClassifier(SGDClassifier(loss="log"), n_jobs=10)
    # log.fit(train_x, train_y)
    # pred_y=log.predict(test_x)
    prec_score=precision_score(test_label,pred_label,average='micro' )
    return prec_score


def generate_10_fold_data(data_length):
    split_interval=int(data_length/10)
    full_index=list(np.arange(data_length))
    Groups=list()
    for i in range(9):
        Groups.append(random.sample(full_index,split_interval))
        full_index = list(set(full_index).difference(set(Groups[i])))
    Groups.append(full_index)
    return Groups

G_MNTED,G_NET,Label=load_data(file_name=file_name_dic[dataset_name])
days=len(G_MNTED)
total_days_score_MNTED=0
total_days_score_NET=0
data=open("classification_result_"+dataset_name+".txt",'a+')
print("Dataset:",dataset_name,file=data)
print("Total time length:",days,file=data)
print("%d x %d fold cross-validation\n"  %(iterateTime , folds),file=data)
data.close()

for day in range(days):
    #num of categories
    print("This is day", day)
    Labels=[]
    for label in Label[day]:
        if label not in Labels:
            Labels.append(label)

    precision_sum_MNTED = 0
    precision_sum_NET = 0
    V_MNTED=G_MNTED[day]
    V_NET=G_NET[day]
    for ite in range(iterateTime):
        print("This is iterate time:",ite)
        class_num=len(Labels)
        train_x = list()
        test_x = list()
        test_index = list()
        train_index=list()
        train_label = list()
        test_label = list()
        bad_flag=False
        Groups=list()
        while not bad_flag:
            print("try to generate suitable Group of folds...")
            Groups = generate_10_fold_data(V_MNTED.shape[0])
            good_flag=list()
            for i in range(10):
                test_index = []
                train_index = []
                train_label = []
                test_label = []
                train_index = Groups[i]
                [test_index.extend(Groups[j]) for j in range(len(Groups)) if j != i]
                [train_label.append(Label[day][k]) for k in train_index]
                [test_label.append(Label[day][k]) for k in test_index]
                flag1=(1 in train_label and 0 in train_label)
                flag2=(1 in test_label and 0 in test_label)
                if flag1 and flag2:
                    good_flag.append(True)
                else:
                    good_flag.append(False)
            if False not in good_flag:
                bad_flag=True
        print("Groups generated successfully!")
        index_1 = Label[day].index(1)
        test_index = []
        train_index = []
        train_label = []
        test_label = []
        for i in range(folds):
            train_index = Groups[i]
            [test_index.extend(Groups[j]) for j in range(len(Groups)) if j != i]
            if index_1 in train_index:
                pass
            else:
                train_index.append(index_1)
                test_index.remove(index_1)
            [train_label.append(Label[day][k]) for k in train_index]
            [test_label.append(Label[day][k]) for k in test_index]
            precision_sum_MNTED += twoLabel_classification(V_MNTED[train_index,], V_MNTED[test_index,],train_label,test_label)
            precision_sum_NET += twoLabel_classification(V_NET[train_index,],V_NET[test_index,],train_label,test_label)
            test_index = []
            train_index = []
            train_label = []
            test_label = []
    day_avg_score_MNTED=precision_sum_MNTED/(iterateTime*folds)
    day_avg_score_NET=precision_sum_NET/(iterateTime*folds)
    total_days_score_MNTED+=day_avg_score_MNTED
    total_days_score_NET+=day_avg_score_NET

    data=open("classification_result_"+dataset_name+".txt",'a+')
    print('MNTED: average score for day %d: %.12f ' %(day, day_avg_score_MNTED),file=data)
    print('NET: average score for day %d: %.12f ' % (day, day_avg_score_NET), file=data)
    data.close()

total_days_avg_score_MNTED=total_days_score_MNTED/days
total_days_avg_score_NET = total_days_score_NET/days
data=open("classification_result_"+dataset_name+".txt",'a+')
print('MNTED: average score for the whole time: %.12f ' %(total_days_avg_score_MNTED),file=data)
print('NET: average score for the whole time: %.12f ' %(total_days_avg_score_NET),file=data)
data.close()
print("classification done!")




from sklearn.svm import SVC
# from sklearn.metrics import classification_report
from sklearn.metrics import precision_score
import numpy as np
import scipy.io as sio
import random
from sklearn.linear_model import SGDClassifier
from sklearn.multioutput import MultiOutputClassifier

file_name_dic = {'aminer': 'aminer/aminer_duicheng',
                 "blog_perturb": 'blog_perturb/blog_perturb',
                 "congress": 'congress/congress_',
                 "Disney": 'Disney/disney_disturb_',
                 "flickr": 'flickr/flickr_disturb_',
                 "wiki": 'wiki/wiki_disturb_'}

'''##############load data################'''
# dataset_name = "aminer"
# dataset_name = "blog_perturb"
dataset_name = "congress"
# dataset_name = "Disney"
# dataset_name = "flickr"
# dataset_name = "wiki"
file_name = file_name_dic[dataset_name]
Label = sio.loadmat(file_name+'8.mat')['gnd']
Label = list(Label.flatten())
embedding_contents = sio.loadmat(file_name+'_Embedding.mat')
V_MNTED=embedding_contents['V_MNTED']
V_Net=embedding_contents['V_Net']

# V_MNTED=np.load(dataset_name+"/"+dataset_name+"-em.npy")
# # V_MNTED=np.load("blog_perturb/blog_perturb-em.npy")

n=V_MNTED.shape[0]
iterateTime=5#cross-validation iterate time
print("data loaded!")
'''##############5-fold cross-validation################'''
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

#num of categories
Labels=[]
for label in Label:
    if label not in Labels:
        Labels.append(label)

embedding_type='V_Net'
# embedding_type="V_MNTED"
precision_sum = 0
for __ in range(iterateTime):

    class_num=len(Labels)
    train_x = list()
    test_x = list()
    test_index = list()
    train_index=list()
    train_label = list()
    test_label = list()
    bad_flag=False
    Groups=list()
    # while not bad_flag:
    #     print("try to generate suitable Group of folds...")
    #     Groups = generate_10_fold_data(V_MNTED.shape[0])
    #     good_flag=list()
    #     for i in range(10):
    #         test_index = []
    #         train_index = []
    #         train_label = []
    #         test_label = []
    #         train_index = Groups[i]
    #         [test_index.extend(Groups[j]) for j in range(len(Groups)) if j != i]
    #         [train_label.append(Label[k]) for k in train_index]
    #         [test_label.append(Label[k]) for k in test_index]
    #         flag1=(1 in train_label and 0 in train_label)
    #         flag2=(1 in test_label and 0 in test_label)
    #         if flag1 and flag2:
    #             good_flag.append(True)
    #         else:
    #             good_flag.append(False)
    #     if False not in good_flag:
    #         bad_flag=True
    #
    #
    # test_index = []
    # train_index=[]
    # train_label = []
    # test_label = []
    # train_x =[]
    # test_x =[]
    #
    # print("Groups generated successfully!")
    Groups = generate_10_fold_data(V_MNTED.shape[0])
    index_1 = Label.index(1)
    for i in range(10):

        train_index = Groups[i]
        [test_index.extend(Groups[j]) for j in range(len(Groups)) if j != i]
        if index_1 in train_index:
            pass
        else:
            train_index.append(index_1)
            test_index.remove(index_1)
        [train_label.append(Label[k]) for k in train_index]
        [test_label.append(Label[k]) for k in test_index]
        if embedding_type == "V_MNTED":
            train_x = V_MNTED[train_index,]
            test_x = V_MNTED[test_index,]
        else:
            train_x = V_Net[train_index,]
            test_x = V_Net[test_index,]
        precision_sum += twoLabel_classification(train_x, test_x,train_label,test_label)
        test_index = []
        train_index = []
        train_label = []
        test_label = []





data=open("result.txt",'a')
print('average precision score of %d times iteration: %.12f for dataset "%s"\n' %(iterateTime , precision_sum/(iterateTime*10),file_name),file=data)
data.close()
print("This is %d x 10 fold cross-validation" %(iterateTime))
print('average precision score of %d times iteration on %s : %.12f for dataset "%s"\n' %(iterateTime ,embedding_type, precision_sum/(iterateTime*10),file_name))

from sklearn.svm import SVC
# from sklearn.metrics import classification_report
from sklearn.metrics import precision_score
import numpy as np
import scipy.io as sio
'''##############load data################'''
# file_name='aminer/aminer_duicheng'
# file_name='blog_perturb/blog_perturb'
# file_name='congress/congress_'
file_name='Disney/disney_disturb_'
# file_name='flickr/flickr_disturb_'
# file_name='wiki/wiki_disturb_'
Label = sio.loadmat(file_name+'8.mat')['best_gnd']
embedding_contents = sio.loadmat(file_name+'_Embedding.mat')
V_MNTED=embedding_contents['V_MNTED']
V_Net=embedding_contents['V_Net']
n=V_MNTED.shape[0]
iterateTime=100#cross-validation iterate time

'''##############5-fold cross-validation################'''
def multiLabel_classification(Labels,Label,Group1,Group2,train_x,test_x):
    Label2=[]
    print("svm")
    score=0
    for label in Labels:
        for i in range(len(Label)):
            if label == Label[i]:
                Label2.append(1)
            else:
                Label2.append(0)
        train_y=[]
        test_y=[]
        for i in Group1:
            train_y.append(Label2[i])
        for i in Group2:
            test_y.append(Label2[i])
        print("set svm")
        clf = SVC(kernel='poly',C=0.4)
        clf.fit(train_x,train_y)
        print("fit done")
        pred_y = clf.predict(test_x)
        print("predict done")
        print('performance of label of ：',label)
        # print(classification_report(test_y,pred_y))
        prec_score=precision_score(test_y,pred_y,average='micro' )
        score=score+prec_score
        Label2=[]
    score=score/len(Labels)
    return score
def twoLabel_classification(Label,Group1,Group2,train_x,test_x):
    train_y = []
    test_y = []
    for i in Group1:
        train_y.append(Label[i])
    for i in Group2:
        test_y.append(Label[i])
    clf = SVC(kernel='poly', C=0.4)
    clf.fit(train_x, train_y)
    pred_y = clf.predict(test_x)
    # print('performance of label of ：', label)
    # print(classification_report(test_y, pred_y))
    prec_score=precision_score(test_y,pred_y,average='micro' )
    return prec_score
def generate_5fold_data(V_MNTED):
    Indices = np.random.randint(100, size=n) + 1  # 5-fold cross-validation indices，size为5196*1的ndarray
    Group1 = []
    Group2 = []
    [Group1.append(x) for x in range(0, n) if Indices[x] <= 90]  # 2 for 10%, 5 for 25%, 20 for 100% of training group 4/5的训练集 随机从n个数里拿4/5n个数放入group，这些数字均小于n
    [Group2.append(x) for x in range(0, n) if Indices[x] >= 91]  # test group 1/5的测试集 随机从n个数里拿1/5n个数放入group，这些数字均小于n
    n1 = len(Group1)  # num of nodes in training group n1大概是n2的4倍
    n2 = len(Group2)  # num of nodes in test group
    # train_x=V_MNTED[Group1+Group2,:][:,Group1+Group2]
    # test_x=V_MNTED[Group2,:][:,Group2]
    V = V_MNTED[Group1 + Group2, :]
    train_x = V_MNTED[0:n1, :]
    test_x = V_MNTED[n1:, :]
    return train_x,test_x,Group1,Group2


precision_sum=0
for __ in range(iterateTime):

    # embedding_type='V_Net'
    embedding_type="V_MNTED"

    if embedding_type=='V_Net':
        train_x, test_x, Group1, Group2 = generate_5fold_data(V_Net)
    else:
        train_x, test_x, Group1, Group2 = generate_5fold_data(V_MNTED)

    #num of categories
    Labels=[]
    for label in Label:
        if label not in Labels:
            Labels.append(label)
    class_num=len(Labels)
    # multiLabel_classification(Labels, Label, Group1, Group2, train_x, test_x)
    if class_num >2:
        precision_sum=precision_sum+multiLabel_classification(Labels,Label,Group1,Group2,train_x,test_x)
    else:
        precision_sum=precision_sum+twoLabel_classification(Label,Group1,Group2,train_x,test_x)



data=open("result.txt",'a')
print('average precision score of %d times iteration: %.12f for dataset "%s"\n' %(iterateTime , precision_sum/iterateTime,file_name),file=data)
data.close()
print('average precision score of %d times iteration on %s : %.12f for dataset "%s"\n' %(iterateTime ,embedding_type, precision_sum/iterateTime,file_name))

from sklearn.svm import SVC
from sklearn.metrics import classification_report
import numpy as np
import scipy.io as sio
'''##############load data################'''
embedding_contents = sio.loadmat('Embedding.mat')
Label = sio.loadmat('BlogCatalog1.mat')['Label']
V_MNTED=embedding_contents['V_MNTED']
V_Net=embedding_contents['V_Net']
n=V_MNTED.shape[0]

'''##############5-fold cross-validation################'''
Indices = np.random.randint(100, size=n) + 1  # 5-fold cross-validation indices，size为5196*1的ndarray
Group1 = []
Group2 = []
[Group1.append(x) for x in range(0, n) if Indices[x] <= 90]  # 2 for 10%, 5 for 25%, 20 for 100% of training group 4/5的训练集 随机从n个数里拿4/5n个数放入group，这些数字均小于n
[Group2.append(x) for x in range(0, n) if Indices[x] >= 91]  # test group 1/5的测试集 随机从n个数里拿1/5n个数放入group，这些数字均小于n
n1 = len(Group1)  # num of nodes in training group n1大概是n2的4倍
n2 = len(Group2)  # num of nodes in test group
# train_x=V_MNTED[Group1+Group2,:][:,Group1+Group2]
# test_x=V_MNTED[Group2,:][:,Group2]
V=V_MNTED[Group1+Group2,:]
train_x=V_MNTED[0:n1,:]
test_x=V_MNTED[n1:,:]
#num of categories
Labels=[]
for label in Label:
    if label not in Labels:
        Labels.append(label)
class_num=len(Labels)
'''################# Classification with SVM #################'''
Label2=[]
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

    clf = SVC(kernel='poly',C=0.4)
    clf.fit(train_x,train_y)
    pred_y = clf.predict(test_x)
    print('performance of label of ：',label)
    print(classification_report(test_y,pred_y))
    Label2=[]

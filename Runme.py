import scipy.io as sio
from MNTED import MNTED
import time

'''################# Load data  #################'''
# file_name='aminer/aminer_duicheng'
# file_name='blog_perturb/blog_perturb'
# file_name='congress/congress_'
# file_name='Disney/disney_disturb_'
# file_name='flickr/flickr_disturb_'
file_name='wiki/wiki_disturb_'
print("Dataset:",file_name)
AttributeMatrixFileName='X'
NetworkMatrixFileName='A'
lambd = 10**-0.6  # the regularization parameter
rho = 5  # the penalty parameter
# mat_contents = sio.loadmat('Flickr.mat')
# lambd = 0.0425  # the regularization parameter
# rho = 4  # the penalty parameter
File_name=file_name+'1'+'.mat'
mat_contents = sio.loadmat(File_name)
i=1
G=[]
A=[]
while mat_contents is not None:
    i=i+1
    Network=mat_contents[NetworkMatrixFileName]
    Attribute=mat_contents[AttributeMatrixFileName]
    # print('the data type of ndarray:',Network.dtype)
    if 'float' not in str(Network.dtype)  or  'double' not in str(Network.dtype)  : Network=Network.astype(float)
    if 'float' not in str(Attribute.dtype)  or  'double' not in str(Attribute.dtype)  : Attribute=Attribute.astype(float)
    G.append(Network)#（5196,5196）
    A.append(Attribute)#（5196,8189）
    # Label = mat_contents["Label"]
    del mat_contents
    n = G[0].shape[0]
    if n>=1000:
        d = 100# the dimension of the embedding representation
    else:
        d=int(n/10)

    # CombG.append(G[Group1+Group2, :][:, Group1+Group2]) #shape：（5196,5196）把原来的G的顺序打乱
    # CombA.append(A[Group1+Group2, :])#shape：（5196,8189）把原来的A的顺序打乱
    try:
        mat_contents = sio.loadmat(file_name+str(i)+'.mat')
    except (FileNotFoundError):
        break








'''################# Multilayer Network Tax Evasion Detection #################'''
print("Multilayer Network Tax Evasion Detection (MNTED), 5-fold with 100% of training is used:")
start_time = time.time()
V_MNTED = MNTED(G, A, d, lambd, rho).function()
print("time elapsed: {:.2f}s".format(time.time() - start_time))

'''################# MNTED for a Pure Network #################'''
print("MNTED for a pure network:")
start_time = time.time()
V_Net = MNTED(G, G, d, lambd, rho).function()
print("time elapsed: {:.2f}s".format(time.time() - start_time))
sio.savemat('Embedding.mat', {"V_MNTED": V_MNTED, "V_Net": V_Net})
print("Embedding.mat printed")






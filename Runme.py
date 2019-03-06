import scipy.io as sio
from MNTED import MNTED
import time


CombG=[]
CombA=[]
'''################# Load data  #################'''
lambd = 10**-0.6  # the regularization parameter
rho = 5  # the penalty parameter
# mat_contents = sio.loadmat('Flickr.mat')
# lambd = 0.0425  # the regularization parameter
# rho = 4  # the penalty parameter

mat_contents = sio.loadmat('BlogCatalog1.mat')
i=1
G=[]
A=[]
while mat_contents is not None:
    '''################# Experimental Settings #################'''
    i=i+1
    d = 100  # the dimension of the embedding representation
    G.append(mat_contents["Network"])#（5196,5196）
    A.append(mat_contents["Attributes"])#（5196,8189）
    # Label = mat_contents["Label"]
    del mat_contents
    n = G[0].shape[0]

    # CombG.append(G[Group1+Group2, :][:, Group1+Group2]) #shape：（5196,5196）把原来的G的顺序打乱
    # CombA.append(A[Group1+Group2, :])#shape：（5196,8189）把原来的A的顺序打乱
    try:
        mat_contents = sio.loadmat('BlogCatalog'+str(i)+'.mat')
    except (FileNotFoundError):
        break








'''################# Accelerated Attributed Network Embedding #################'''
print("Accelerated Attributed Network Embedding (AANE), 5-fold with 100% of training is used:")
start_time = time.time()
V_MNTED = MNTED(G, A, d, lambd, rho).function()
print("time elapsed: {:.2f}s".format(time.time() - start_time))

'''################# AANE for a Pure Network #################'''
print("AANE for a pure network:")
start_time = time.time()
V_Net = MNTED(G, G, d, lambd, rho).function()
print("time elapsed: {:.2f}s".format(time.time() - start_time))
sio.savemat('Embedding.mat', {"V_MNTED": V_MNTED, "V_Net": V_Net})
print("Embedding.mat printed")






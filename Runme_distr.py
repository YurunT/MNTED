import scipy.io as sio
from MNTED_distr import MNTED
import time

'''################# Load data  #################'''
file_name_dic = {'aminer': 'aminer/aminer_duicheng',
                 "blog_perturb": 'blog_perturb/blog_perturb',
                 "congress": 'congress/congress_',
                 "Disney": 'Disney/disney_disturb_',
                 "flickr": 'flickr/flickr_disturb_',
                 "wiki": 'wiki/wiki_disturb_'}
# dataset_name = "aminer"
dataset_name = "blog_perturb"
# dataset_name = "congress"
# dataset_name = "Disney"
# dataset_name = "flickr"
# dataset_name = "wiki"
file_name=file_name_dic[dataset_name]
print("Dataset:",dataset_name)
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
if __name__ == "__main__":
    worker_num=2
    split_num=6
    print("MNTED for a attribute network")
    start_time = time.time()
    V_MNTED = MNTED(G, A, d, lambd, rho,4,'Attr',worker_num,split_num).function()
    V_MNTED_time_worker=time.time() - start_time
    print("time elapsed with {:d} worker: {:.2f}s".format(worker_num,V_MNTED_time_worker))


    '''################# MNTED for a Pure Network #################'''
    print("MNTED for a pure network:")
    start_time = time.time()
    V_Net = MNTED(G, G, d, lambd, rho,4,'Net',worker_num,split_num).function()
    V_NET_time_worker=time.time() - start_time
    print("time elapsed with {:d} worker: {:.2f}s".format(worker_num,V_NET_time_worker))


    for i in range(len(V_MNTED)):
        sio.savemat(file_name+'_Embedding'+str(i)+"_"+str(worker_num)+'workers.mat', {"V_MNTED": V_MNTED[i], "V_Net": V_Net[i]})
    print("Embedding.mat printed")


    data=open(file_name+"_"+"time_result"+"_"+str(worker_num)+"workers"+".txt",'a+')
    print("Dataset:",dataset_name,file=data)
    print("time length:", len(V_MNTED), file=data)
    print("Time spent on MNTED: %.12f s with %d workers"  %(V_NET_time_worker,worker_num),file=data)
    print("Time spent on NET: %.12f s with %d workers"  %(V_NET_time_worker,worker_num),file=data)
    data.close()
    print("All process finished!")
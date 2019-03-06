import numpy as np
from scipy.sparse import csc_matrix
G=np.array([[1.1,1.2,1.3,1.4],
            [2.1,2.2,2.3,2.4],
            [3.1,3.2,3.3,3.4],
            [4.1,4.2,4.3,4.4]])
G2=csc_matrix(G)
Group1=[1,3]
Group2=[0,2]
print(G)
print("###########")
print(G2[Group1+Group2, :][:, Group1+Group2].toarray())
# print(G2[Group1+Group2, :].toarray())
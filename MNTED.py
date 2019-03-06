import numpy as np
from scipy import sparse
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import svds
from math import ceil

class MNTED:
    """
    :param  MultiNet: a set of the weighted adjacency matrices（a list of ndarray(n*n)）
    :param  MultiAttri: a set of the weighted adjacency matrices （a list of ndarray(n*m)）
    :param  d: the dimension of the embedding representation
    :param  n: the num of nodes per layer
    :param  m: the dimension of the attribute
    :param  k: the num of layers in total
    :param  lambd: the regularization parameter
    :param  rho: the penalty parameter
    :param  maxiter: the maximum number of iteration
    :param  'Att': refers to conduct Initialization from the SVD of Attri
    :param  splitnum: the number of pieces we split the SA for limited cache
    :return the comprehensive embedding representation H
    """
    def __init__(self, MultiNet, MultiAttri, d, *varargs):
        '''

        :param MultiNet: a list  of network matrices with shape of (n,n)
        :param MultiAttri: a list of attribute matrices with shape of (n,m)
        :param d: the dimension of the embedding representation
        :param varargs: 0：lambd，1：rho，2：maxiter，3：Att, 4:splitnum
        :returns initialization of multiple core variable
        '''
        self.maxiter = 2  # Max num of iteration
        self.lambd = 0.05  # Initial regularization parameter
        self.rho = 5  # Initial penalty parameter
        self.k=len(MultiNet)#the num of layers of Multilayer Network
        print('k:',self.k)
        self.d = d
        splitnum = 1  # number of pieces we split the SA for limited cache
        [self.n, m] = MultiAttri[0].shape  # n = Total num of nodes, m = attribute category num
        print('len(MultiNet):',len(MultiNet))
        print('MultiNet[0].shape:',MultiNet[0].shape)
        Nets=[]
        Attris=[]
        for Net in MultiNet:
            Net = sparse.lil_matrix(Net)
            Net.setdiag(np.zeros(self.n))
            Net = csc_matrix(Net)  # 在用python进行科学运算时，常常需要把一个稀疏的np.array压缩
            Nets.append(Net)
        # Nets=[csc_matrix(sparse.lil_matrix(Net).setdiag(np.zeros(self.n))) for Net in MultiNet]
        for Attri in MultiAttri:
            Attri=csc_matrix(Attri)
            Attris.append(Attri)
        # Attris=[csc_matrix(Attri) for Attri in MultiAttri]
        self.H=[]
        if len(varargs) >= 4 and varargs[3] == 'Att':
            # 将属性矩阵A打乱成n*m（或n*10d）的新矩阵再进行svd分解后的酉矩阵，规格为n*d，作为H的初始值
            for Atrri in Attris:
                sumcol = np.arange(m)# [0,1,...,m-1]
                np.random.shuffle(sumcol)# 打乱
                self.H.append(svds(Atrri[:, sumcol[0:min(10 * d, m)]], d)[0])
        else:
            # 将拓扑矩阵打乱成n*n或（n*10d）的新矩阵（按照纵向求和的从大到小排列），再进行svd分解后的酉矩阵，规格为n*d，作为H的初始值
            for Net in Nets:
                print(type(Net))
                print(Net.shape)
                sumcol = Net.sum(0)#将Net沿纵方向向下加，成为一个长度为n的向量
                H_initial=svds(Net[:, sorted(range(0,self.n), key=lambda r: sumcol[0, r], reverse=True)[0:min(10 * d, self.n)]], d)[0]
                self.H.append(H_initial)
                # svds(Net[:, sorted(range(self.n), key=lambda r: sumcol[0, r], reverse=True)[0:min(10 * d, self.n)]], d)[0]
        if len(varargs) > 0:
            self.lambd = varargs[0]
            self.rho = varargs[1]
            if len(varargs) >= 3:
                self.maxiter = varargs[2]
                if len(varargs) >= 5:
                    splitnum = varargs[4]
        self.block = min(int(ceil(float(self.n) / splitnum)), 7575)  # Treat at least（most？？） each 7575 nodes as a block。即将n个节点分成splitnum个block，1个block最多有7575个节点
        self.splitnum = int(ceil(float(self.n) / self.block))#重新计算splitnum
        with np.errstate(divide='ignore'):  # inf will be ignored,即不管异常与否，都会进行下面的计算
            self.Attri = [Attri.transpose() * sparse.diags(np.ravel(np.power(Attri.power(2).sum(1), -0.5))) for Attri in Attris]#计算属性矩阵
        self.Z = self.H.copy()
        self.affi = -1  # Index for affinity matrix sa
        self.U = [np.zeros((self.n, d)) for i in range(self.k)]#U是n*d的全0矩阵
        self.V=self.H[0]#V的初始值和H的第一个图的初始值一样
        self.nexidx = [np.split(Net.indices, Net.indptr[1:-1]) for Net in
                       Nets]  # 将每一列的非零元素的坐标分开，得到A list of sub-arrays.
        self.Net = [np.split(Net.data, Net.indptr[1:-1]) for Net in Nets]  # 将每一列的非零数据分开，得到A list of sub-arrays


    '''################# Update functions #################'''
    def updateH(self,k):
        xtx = np.dot(self.Z[k].transpose(), self.Z[k]) * 2 + (2+self.rho) * np.eye(self.d)
        for blocki in range(self.splitnum):  # Split nodes into different Blocks
            indexblock = self.block * blocki  # Index for splitting blocks
            if self.affi != blocki:
                self.sa = self.Attri[k][:, range(indexblock, indexblock + min(self.n - indexblock, self.block))].transpose() * self.Attri[k]
                self.affi = blocki
            sums = self.sa.dot(self.Z[k]) * 2
            for i in range(indexblock, indexblock + min(self.n - indexblock, self.block)):
                neighbor = self.Z[k][self.nexidx[k][i], :]  # the set of adjacent nodes of node i
                for j in range(1):
                    normi_j = np.linalg.norm(neighbor - self.H[k][i, :], axis=1)  # norm of h_i^k-z_j^k
                    nzidx = normi_j != 0  # Non-equal Index
                    if np.any(nzidx):
                        normi_j = (self.lambd * self.Net[k][i][nzidx]) / normi_j[nzidx]
                        self.H[k][i, :] = np.linalg.solve(xtx + normi_j.sum() * np.eye(self.d), sums[i - indexblock, :] + (
                                    neighbor[nzidx, :] * normi_j.reshape((-1, 1))).sum(0) -(2-self.rho)*self.Z[k][i,:]+4*self.V[i,:]-self.rho*self.U[k][i,:])
                    else:
                        self.H[k][i, :] = np.linalg.solve(xtx, sums[i - indexblock, :] -(2-self.rho)*self.Z[k][i,:]+4*self.V[i,:]-self.rho*self.U[k][i,:])
    def updateZ(self,k):
        xtx = np.dot(self.H[k].transpose(), self.H[k]) * 2 + (2+self.rho) * np.eye(self.d)
        for blocki in range(self.splitnum):  # Split nodes into different Blocks
            indexblock = self.block * blocki  # Index for splitting blocks
            if self.affi != blocki:
                self.sa =self.Attri[k][:, range(indexblock, indexblock + min(self.n - indexblock, self.block))].transpose() * self.Attri[k]
                self.affi = blocki
            sums = self.sa.dot(self.H[k]) * 2
            for i in range(indexblock, indexblock + min(self.n - indexblock, self.block)):
                neighbor = self.H[k][self.nexidx[k][i], :]  # the set of adjacent nodes of node i
                for j in range(1):
                    normi_j = np.linalg.norm(neighbor - self.Z[k][i, :], axis=1)  # norm of h_i^k-z_j^k
                    nzidx = normi_j != 0  # Non-equal Index
                    if np.any(nzidx):
                        normi_j = (self.lambd * self.Net[k][i][nzidx]) / normi_j[nzidx]
                        self.Z[k][i, :] = np.linalg.solve(xtx + normi_j.sum() * np.eye(self.d), sums[i - indexblock, :] + (
                                    neighbor[nzidx, :] * normi_j.reshape((-1, 1))).sum(0)
                                    - (2-self.rho)*self.H[k][i,:]+4*self.V[i,:]-self.rho*self.U[k][i,:])
                    else:
                        self.Z[i, :] = np.linalg.solve(xtx, sums[i - indexblock, :]
                                    - (2-self.rho)*self.H[k][i,:]+4*self.V[i,:]-self.rho*self.U[k][i,:])
    def updateV(self,k):
        self.V=1/2*(self.H[k]+self.Z[k])
    def function(self):
        '''################# Iterations #################'''
        for i in range(self.k):
            for itr in range(self.maxiter - 1):
                self.updateH(i)
                self.updateZ(i)
                self.updateV(i)
                self.U[i]=self.U[i] + self.H[i] - self.Z[i]
        return self.Vav


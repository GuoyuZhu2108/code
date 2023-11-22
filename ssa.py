import copy
import random
import numpy as np
import math as m
from env import Point2D
import matplotlib.pyplot as plt
import numpy as np

uav = {}
neigh_uav = {}
num = 7




def draw(next_x ,next_y):
    # 表格风格
    x = [0]*7
    y = [0]*7
    for i in range (num):
        x[i] = uav[i].x
        y[i] = uav[i].y

    plt.scatter(x, y,s = 20, marker = '.',c = 'blue',label='uav')
    x = [0]*len(neigh_uav)
    y = [0]*len(neigh_uav)
    for i in range(len(neigh_uav)):
        x[i] = neigh_uav[i].x
        y[i] = neigh_uav[i].y
    plt.scatter(x, y, s=20, marker='.',c = 'green',label='neighbor_uav')
    plt.scatter(0, 0, s=50,marker='o',c = 'black',label='lost_uav')

    plt.scatter(next_x, next_y, s=100,marker='x',c = 'red',label='Next position')





def sigmoid(x):

    y=1/(1+np.exp(-x))
    dy=y*(1-y)
    return dy





class SSA:
    ''' Tent种群初始化函数 '''

    def __init__(self, pop, dim, ub, lb ,neighu,lost_uav,obs):
        self.X = np.zeros([pop, dim])
        self.pop = pop
        self.dim = dim
        self.ub = ub
        self.lb = lb
        self.neighu = neighu
        self.lost_uav = lost_uav
        self.obs = obs

        for i in range(self.pop):
            self.X[i, 0] = np.random.rand() * (self.ub[0] - self.lb[0]) + self.lb[0]+ self.lost_uav.x
            self.X[i, 1] = np.random.rand() * (self.ub[1] - self.lb[1]) + self.lb[1]+ self.lost_uav.y



    '''边界检查函数'''


    def BorderCheck(self):
        for i in range(self.pop):

            while ((self.X[i,0]-self.lost_uav.x) * (self.X[i,0]-self.lost_uav.x) + (self.X[i,1]-self.lost_uav.y) * (self.X[i,1]-self.lost_uav.y)) > 100:
                self.X[i, 0] = np.random.rand() * (self.ub[0] - self.lb[0]) + self.lb[0] + self.lost_uav.x
                self.X[i, 1] = np.random.rand() * (self.ub[1] - self.lb[1]) + self.lb[1] + self.lost_uav.y
        return self.X


    '''定义适应度函数'''
    def funa(self,a):
        r = 0
        neigh_num = len(self.neighu)
        l = [float(0)] * neigh_num

        for i in range(neigh_num):
            j = (self.neighu[i].x-a[0])*(self.neighu[i].x-a[0])
            k = (self.neighu[i].y-a[1])*(self.neighu[i].y-a[1])
            l[i] = j + k#计算距离
        for i in range(4):
            j = (self.obs[i].x - a[0]) * (self.obs[i].x - a[0])
            k = (self.obs[i].y - a[1]) * (self.obs[i].y - a[1])
            rew = 0
            if(j+k)<900:
                rew = -25
            r = r + rew
        for i in range (neigh_num):
            if(type(l[i]) == complex):
                continue

            l[i] = m.sqrt(l[i])

            if(l[i] < 3):
                rew = -20
            if(l[i]>=3 and l[i]<=20):
                rew = 20

            if (l[i] > 15):
                rew = -20

            r = r +rew
        rew = 32*m.sqrt((self.lost_uav.x-a[0])*(self.lost_uav.x-a[0])+(self.lost_uav.y-a[1])*(self.lost_uav.y-a[1]))
        r = r - rew
        r = r/neigh_num
        return r

    '''计算适应度函数'''


    def CaculateFitness(self):
        pop = self.X.shape[0]
        fitness = np.zeros([pop, 1])
        for i in range(pop):
            fitness[i] = SSA.funa(self,self.X[i])
        return fitness


    '''适应度排序'''


    def SortFitness(self,Fit):
        self.fitness = np.sort(-Fit, axis=0)
        self.index = np.argsort(-Fit, axis=0)
        self.fitness = -self.fitness
        return self.fitness, self.index


    '''根据适应度对位置进行排序'''


    def SortPosition(self):
        Xnew = np.zeros(self.X.shape)
        for i in range(self.X.shape[0]):
            Xnew[i, :] = self.X[self.index[i], :]
        return Xnew


    '''麻雀发现者勘探更新'''


    def PDUpdate(self, PDNumber, ST, Max_iter):
        X_new = copy.copy(self.X)
        R2 = random.random()
        for p in range(PDNumber):
            for j in range(self.dim):
                if R2 < ST:
                    X_new[p, j] = self.X[p, j] * np.exp(-p / (random.random() * Max_iter))
                else:
                    X_new[p, j] = self.X[p, j] + np.random.randn()
        return X_new


    '''麻雀加入者更新'''


    def JDUpdate(self, PDNumber):
        X_new = copy.copy(self.X)
        A = np.ones([self.dim, 1])
        for a in range(self.dim):
            if (random.random() > 0.5):
                A[a] = -1
        for i in range(PDNumber + 1, self.pop):
            for j in range(self.dim):
                if i > (self.pop - PDNumber) / 2 + PDNumber:
                    X_new[i, j] = np.random.randn() * np.exp((self.X[-1, j] - self.X[i, j]) / i ** 2)
                else:
                    AA = np.mean(np.abs(self.X[i, :] - self.X[0, :]) * A)
                    X_new[i, j] = self.X[0, j] - AA
        return X_new


    '''危险更新'''


    def SDUpdate(self, SDNumber, BestF):
        X_new = copy.copy(self.X)
        dim = self.X.shape[1]
        Temp = range(self.pop)
        RandIndex = random.sample(Temp, self.pop)
        SDchooseIndex = RandIndex[0:SDNumber]
        for i in range(SDNumber):
            for j in range(dim):
                if self.fitness[SDchooseIndex[i]] > BestF:
                    X_new[SDchooseIndex[i], j] = self.X[0, j] + np.random.randn() * np.abs(self.X[SDchooseIndex[i], j] - self.X[0, j])
                elif self.fitness[SDchooseIndex[i]] == BestF:
                    K = 2 * random.random() - 1
                    X_new[SDchooseIndex[i], j] = self.X[SDchooseIndex[i], j] + K * (
                            np.abs(self.X[SDchooseIndex[i], j] - self.X[-1, j]) / (self.fitness[SDchooseIndex[i]] - self.fitness[-1] + 10E-8))
        return X_new


    '''麻雀搜索算法'''


def Tent_SSA(pop, dim, lb, ub, Max_iter,neigh,losuav,obs):
    ST = 0.6  # 预警值
    PD = 0.7  # 发现者的比列，剩下的是加入者
    SD = 0.2  # 意识到有危险麻雀的比重
    PDNumber = int(pop * PD)  # 发现者数量
    SDNumber = int(pop * SD)  # 意识到有危险麻雀数量
    S = SSA(pop, dim, ub, lb,neigh,losuav,obs)# 初始化种群
    S.BorderCheck()
    fitness = S.CaculateFitness()  # 计算适应度值
    fitness, sortIndex = S.SortFitness(fitness)  # 对适应度值排序
    X = S.SortPosition()  # 种群排序
    GbestScore = copy.copy(fitness[0])
    GbestPositon = np.zeros([1, dim])
    GbestPositon[0, :] = copy.copy(X[0, :])
    Curve = np.zeros([Max_iter, 1])
    for i in range(Max_iter):
        BestF = fitness[0]

        X = S.PDUpdate(PDNumber, ST, Max_iter)  # 发现者更新

        X = S.JDUpdate(PDNumber)  # 加入者更新

        X = S.SDUpdate(SDNumber, BestF)  # 危险更新

        X = S.BorderCheck()  # 边界检测

        fitness = S.CaculateFitness()  # 计算适应度值

        fitness, sortIndex = S.SortFitness(fitness)  # 对适应度值排序
        X = S.SortPosition()  # 种群排序
        if (fitness[0] >= GbestScore):  # 更新全局最优
            GbestScore = copy.copy(fitness[0])
            GbestPositon[0, :] = copy.copy(X[0, :])
        Curve[i] = GbestScore
    return GbestPositon
step = 18


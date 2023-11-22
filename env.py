import math as m
import random
import numpy as np
from matplotlib import pyplot as plt

class Point2D:
    def __init__(self):
        self.x = complex()
        self.y = complex()

class ControlInfo:
    def __init__(self):
        self.c = [None]*2


class Observation:
    def __init__(self):
        self.o= [None] * 48

class pair:
    def __init__(self):
        self.first = 0
        self.second = 0



def l2norm (x,y):
    return m.sqrt(x * x + y * y)

def sigmoid(x):
    y=1/(1+np.exp(-x))
    dy=y*(1-y)
    return dy

def getDictKey_1(myDict, value):
     for k, v in myDict.items():
         if v == value:
             break
     return k

DIE_PENALTY = 100
UAV_COLLISION_R = 3
PI=3.14159265358979323846
UAV_UAV_COMM = 50
m_obstacless = {}
OBSTACLE_CNT = 0
m_target = Point2D()
m_die = []
m_status = []
in_target_area = 0
collision_with_obs = 0
collision_with_uav = 0
m_prev_pos = {}
m_next_pos = {}
m_obstacles = []
m_collision = []
die_on_this_step = []
MAX_DIST_PENALTY = 50
MAX_DIST_RANGE = 500
TARGET_R = 50
TASK_REW = 100

def getStatus(self):
    return m_die

class UAVModel2D:

    def __init__(self, x, y, v, w):
        self.m_x = x
        self.m_y = y
        self.m_v = v
        self.m_w = w

    def step(self, ang, acc):

        self.m_w += ang
        self.m_v += acc
        if (self.m_v > 6) :
            self.m_v = 6
        if (self.m_v < 2) :
            self.m_v = 2
        while (self.m_w > 3*PI/4) :
            self.m_w = 3*PI/4
        while (self.m_w < -PI/4) :
            self.m_w = 0

        self.m_x += self.m_v * m.cos(self.m_w)
        self.m_y += self.m_v * m.sin(self.m_w)

m_uavs = {}
start = {}



class ManyUavEnv:
    def __init__(self, uav_cnt, random_seed,uav_die = True):
        self.uav_die = uav_die
        self.m_uav_cnt = uav_cnt
        self.m_rnd_engine = random_seed
        self.m_target = {}
        self.m_steps = 0
        self.succ_cnt = 0
        self.sumway = [0.0]*uav_cnt
        self.endstep = [0.0]*uav_cnt

    def getStatus(self):
        return m_die

    def reset(self):
        m_uavs.clear()
        m_uavs.clear()
        m_obstacles.clear()
        k=0
        a = 49
        for i in range (self.m_uav_cnt) :
            if(k == 3):
                k = 0
            if(i % 3 ==0):
                a = a - 3
            k = k + 1

            m_uavs[i] = UAVModel2D(9+3*k,a , 2.0, PI/4)

            start[i] = Point2D()
            start[i] .x = m_uavs[i].m_x
            start[i] .y = m_uavs[i].m_y

        m_obstacless[3] = Point2D()
        m_obstacless[3].x= 200
        m_obstacless[3].y = 200

        m_obstacless[2] = Point2D()
        m_obstacless[2].x = 350
        m_obstacless[2].y = 320

        m_obstacless[1] = Point2D()
        m_obstacless[1].x = 200
        m_obstacless[1].y = 320

        m_obstacless[0] = Point2D()
        m_obstacless[0].x = 300
        m_obstacless[0].y = 100


        m_target.x = 475
        m_target.y = 475
        m_steps = 0
        m_die.clear()
        m_status.clear()
        for i in range (self.m_uav_cnt):
            m_die.append(False)
            m_status.append(0)
        self.succ_cnt = 0

    def step(self,control):
        self.in_target_area = 0
        self.collision_with_uav = 0
        m_prev_pos.clear()
        m_next_pos.clear()

        for i in range (self.m_uav_cnt):
            if (m_die[i]) :continue
            m_prev_pos[i] = [m_uavs[i].m_x, m_uavs[i].m_y]
            m_uavs[i].step(control[i][0], control[i][1])
            m_next_pos[i] = [m_uavs[i].m_x, m_uavs[i].m_y]
            #启发式在这改 改变速度和角速度

        self.m_steps = self.m_steps + 1

        if self.m_steps == 200:
            self.m_steps = self.m_steps - 200

    def getObservations(self):
        result = []
        center = Point2D()
        center.x = 0
        center.y = 0
        cnt = 0
        for i in range(self.m_uav_cnt):
            if m_status[i] != 1:
                center.x += m_uavs[i].m_x
                center.y += m_uavs[i].m_y
                cnt += 1
        if cnt != 0:
            center.x /= cnt
            center.y /= cnt
        for i in range(self.m_uav_cnt):
            obs = Observation().o
            obs[0] = (m_uavs[i].m_x) / 500.
            obs[1] = (m_uavs[i].m_y) / 500.
            obs[2] = (m_uavs[i].m_v - 2) / 4.
            obs[3] = (m_uavs[i].m_w+PI/4) / (PI)
            index_dist = {}
            for j in range (self.m_uav_cnt):
                if (j == i) : continue
                if(m_die[j]): continue
                dist = l2norm((m_uavs[j].m_x - m_uavs[i].m_x), (m_uavs[j].m_y - m_uavs[i].m_y ))
                index_dist[j] = dist


            index_dist = sorted(index_dist.items(),  key=lambda d: d[1], reverse=False)
            index_dist = dict(index_dist)#字典
            value = sorted(index_dist.values())#value
            for j in range (8):
                if j < len(index_dist) and value[j] < UAV_UAV_COMM:
                    key = getDictKey_1(index_dist, value[j])
                    obs[4 + j * 4] = (m_uavs[key].m_x) / 500.
                    obs[5 + j * 4] = (m_uavs[key].m_y) / 500.
                    obs[6 + j * 4] = (m_uavs[key].m_v - 2) / 4.
                    obs[7 + j * 4] = (m_uavs[key].m_w+PI/4) / (PI)
                else:
                    obs[4 + j * 4] = -1.0
                    obs[5 + j * 4] = -1.0
                    obs[6 + j * 4] = 0.0
                    obs[7 + j * 4] = 0.0
            for k in range (4):
                obs[36+k] = (m_obstacless[k].x ) / 500
            for k in range (4):
                obs[40+k] = (m_obstacless[k].y) / 500
            obs[44] = (m_target.x - 450.) / 50.
            obs[45] = (m_target.y - 450.) / 50.
            obs[46] = int(m_die[i])
            obs[47] = self.m_steps / 200.
            result.append(obs)
        return np.array(result)

    def getRewards(self):
        global collisiono
        result = [0.0]*self.m_uav_cnt

        for i in range (self.m_uav_cnt):
            if(m_die[i]):
                result[i] = 0
                continue

            self.sumway[i] = 0

            result[i] = result[i] - 2
            dp = l2norm(m_prev_pos[i][0] - m_target.x, m_prev_pos[i][1] - m_target.y)
            dn = l2norm(m_next_pos[i][0] - m_target.x, m_next_pos[i][1] - m_target.y)
            dstep = l2norm(m_next_pos[i][0]-m_prev_pos[i][0],m_next_pos[i][1]-m_prev_pos[i][1])
            result[i] = 1.1*(dp - dn) +result[i]

            self.sumway[i] = dstep

            die_on_this_step =[False]*self.m_uav_cnt


            for j in range(self.m_uav_cnt):
                if(m_die[j] or i == j):continue
                if(l2norm(m_next_pos[j][0] - m_next_pos[i][0], m_next_pos[j][1] - m_next_pos[i][1]) < 3):
                    result[i] -= 15
                    break

            #避障
            collisiono = False
            for k in range (4):
                if(l2norm(m_next_pos[i][0] - m_obstacless[k].x, m_next_pos[i][1] - m_obstacless[k].y) < 30+k*5):
                    collisiono = True
                    break
            if(collisiono):
                result[i] -= 75
                die_on_this_step[i] = True

            for k in range (4):
                if(k>1):
                    if(l2norm(m_next_pos[i][0] - m_obstacless[k].x, m_next_pos[i][1] - m_obstacless[k].y) > 30+k*5 and l2norm(m_next_pos[i][0] - m_obstacless[k].x, m_next_pos[i][1] - m_obstacless[k].y) < 40+k*5):
                        result[i] -= 2


            #群集
            for j in range (self.m_uav_cnt):
                if(m_die[j]): continue
                if i != j :
                    dist = l2norm(m_next_pos[j][0]- m_next_pos[i][0], m_next_pos[j][1] - m_next_pos[i][1])
                    if (dist < 20 and dist > 3) :
                        result[i] += 0.7

            if (l2norm(m_next_pos[i][0] - m_target.x, m_next_pos[i][1] - m_target.y) < 20) :
                result[i] += 150
                if (die_on_this_step[i] == False):
                    self.endstep[i] = self.m_steps
                die_on_this_step[i] = True

        for i in range (self.m_uav_cnt) :
            if die_on_this_step[i]:
                m_die[i] = True

        return result

    def isDone(self):
        all_die = True
        for i in range (self.m_uav_cnt) :
            if (m_die[i] == False) :all_die = False

        return (self.m_steps % 200 == 0) or all_die

    def getObstacles(self):
        return m_obstacless

    def getUavs(self):
        return m_next_pos

    def getCollision(self):
        return m_collision

    def getTarget(self):
        return m_target

    def getWay(self):
        return self.sumway

    def getEnd(self):
        return self.endstep

    def getuavW(self):
        w = [0.0]*self.m_uav_cnt
        for i in range (self.m_uav_cnt):
            w[i] = m_uavs[i].m_w
        return w
    def getuavV(self):
        v = [0.0] * self.m_uav_cnt
        for i in range(self.m_uav_cnt):
            v[i] = m_uavs[i].m_v
        return v




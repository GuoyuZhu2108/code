import os
import csv
import numpy as np
from matplotlib import pyplot as plt


def drawback():
    fig = plt.figure(figsize=(8, 8))
    ax = plt.axis([0, 500, 0, 500])


def drawtarget(x,y):
    plt.text(x - 18, y - 30, 'Target')
    r = 20.0
    # 2.圆心坐标
    a = x
    b = y
    # ==========================================
    # 方法一：参数方程
    theta = np.arange(0, 2 * np.pi, 0.01)
    x = a + r * np.cos(theta)
    y = b + r * np.sin(theta)
    plt.plot(x, y,linestyle='solid',c = 'black')
    plt.legend(fontsize=10, loc="lower right")



def savepic(now_ep):
    figure_save_path = "file_fig"
    if not os.path.exists(figure_save_path):
        os.makedirs(figure_save_path)
    plt.savefig('./file_fig/pic-{}.png'.format(now_ep))


def draw_uav(x,y,u):
    p1 = plt.scatter(x, y, s=8, marker='.', c=u)

def drawobs(x, y, i):
    r = 28.0+i*5
    i =i +1
    if (i < 3):
        plt.scatter(x, y, s=r*2*150*(0.23*i+1.2), marker='.', c='gray')
    if (i > 2):
        plt.scatter(x, y, s=r*2*150*(0.23*i+1.2), marker='.', color=(254/255,232/255,154/255))
    if (i < 3):
        plt.text(x-45, y-1.5*r,'Obstacles',fontsize=15)
    if (i > 2):
        plt.text(x - 85, y - 1.79 * r, 'Interference source', fontsize=15)
    if( i >2):
    # 2.圆心坐标
        a = x
        b = y
        # ==========================================
        # 方法一：参数方程
        n = 35.0+i*5
        theta = np.arange(0, 2 * np.pi, 0.01)
        x = a + n * np.cos(theta)
        y = b + n * np.sin(theta)
        # axes = fig.add_subplot(111)
        plt.plot(x, y,linestyle='solid',c = 'orange' , linewidth=2)

def drawnext(next_x ,next_y):
    plt.scatter(next_x, next_y, s=10,marker='*',c = 'red',label='Next position')


def newcsv(sumway,End,Rew,f,lostend,lostway,losttime):

    # 2. 基于文件对象构建 csv写入对象
    csv_writer = csv.writer(f)

    # 3. 构建列表头
    csv_writer.writerow(sumway)

    csv_writer.writerow(End)

    csv_writer.writerow(Rew)

    csv_writer.writerow([lostend])

    csv_writer.writerow([lostway])

    csv_writer.writerow([losttime])



def drawUAV(uav,uav_num):
    for u in range (uav_num):
        draw_uav(uav[u][0], uav[u][1],(204/255,103/255,102/255))


if __name__ == '__main__':
    main()
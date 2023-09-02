from datetime import datetime

import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt

import time
import cv2
import random

def ts2num(t):
    h,m,s = t.strip().split(':')
    n = int(h)*3600 + int(m)*60 + int(s)
    return n

def num2ts(n):
    h = n // 3600 #t.split(:)
    rs = n % 3600 
    m = rs // 60
    s = rs % 60
    
    return h, m, s

if __name__ == "__main__"

    l = [0,1,2]

    start_time = datetime.now().time().strftime("%H:%M:%S") # strftime("%Y-%m-%d %H:%M:%S")
    print(start_time)


    num_mins = 5
    start_n = ts2num(start_time)
    window_duration = num_mins*60

    print(start_n, window_duration)

    fig = plt.figure(figsize=(10,3))
    plt.rcParams.update({'font.size': 14})
    plt.yticks([], [])
    plt.ylim([0,1])
    plt.xlim([start_n, start_n + window_duration])
    plt.xlabel('Time stamp (HH:MM)')
    plt.title('TV viewing behavior')

    label_xticks = []
    for i in range(num_mins):
        print(i, start_n + i*num_mins*60)
        h,m,s = num2ts(start_n + i*60)
        
        k = '%02d:%02d'%(h,m)
        label_xticks.append(k)


    plt.xticks(ticks=range(start_n,start_n+window_duration,60),labels=label_xticks)

    for i in range(1000):
        random.shuffle(l)
        val = l[0:1][0]
        
        now_time = datetime.now().time().strftime("%H:%M:%S")
        n_now = ts2num(now_time)
        h,m,s = num2ts(n_now)
        print(now_time, val)

        if val==2:
            time.sleep(0.3)
            continue
            
        colors = 'yellowgreen' if l[0]==1 else 'deepskyblue'
            
        plt.bar(n_now, 1, color=colors, width=1)
        plt.pause(0.01)
        time.sleep(0.3)
        
        if start_n + window_duration == n_now: #(n_now - 1) or start_n + window_duration == (n_now + 1):
            plt.clf()
            start_n = n_now
            plt.xlim([start_n, start_n + window_duration])


    plt.show()

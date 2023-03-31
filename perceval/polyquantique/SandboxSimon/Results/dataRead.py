import os 
import numpy as np
from matplotlib import pyplot as plt 
import json 
import tabulate as tab



def txtRead(filename):
    time = []
    count = []
    with open(filename,'r') as file:
        file.readline()
        for line in file.readlines():
            time.append(float(line.split(',')[0]))
            count.append(float(str.strip((line.split(',')[1]),'\n')))
    return np.array(time),np.array(count)

def xpRead(filename):
    with open(filename) as file:
        data = json.load(file)
        time = data["delay data"]['ch2']['x']
        env1 = data["delay data"]['ch2']['y'] 
        env2 = data["delay data"]['ch4']['y']
        delay = data["delay data"]["delay"]
        c00 = data['cnot']["00"]
        c01 = data['cnot']["01"]
        c10 = data['cnot']["10"]
        c11 = data['cnot']["11"]
        tot00 = sum(c00.values())
        probsc00 = np.divide([c00["00"], c00["01"],c00["10"],c00["11"]],tot00)
        tot01 = sum(c01.values())
        probsc01 = np.divide([c01["00"], c01["01"],c01["10"],c01["11"]],tot01)
        tot10 = sum(c10.values())
        probsc10 = np.divide([c10["00"], c10["01"],c10["10"],c10["11"]],tot10)
        tot11 = sum(c11.values())
        probsc11 = np.divide([c11["00"], c11["01"],c11["10"],c11["11"]],tot11)
        table = np.array([probsc00,probsc01,probsc10,probsc11])


        

    return time, env1, env2,delay, table
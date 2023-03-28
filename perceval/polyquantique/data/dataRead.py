import numpy as np
import json

def txtRead(filename):
    time = []
    count = []
    with open(filename,'r') as file:
        file.readline()
        for line in file.readlines():
            time.append(float(line.split(',')[0]))
            count.append(float(str.strip((line.split(',')[1]),'\n')))
    return np.array(time),np.array(count)

def jsonRead(filename):
    with open(filename) as file:
        data = json.load(file)
        time = data["delay data"]['ch3']['x']
        env1 = data["delay data"]['ch3']['y'] 
        env2 = data["delay data"]['ch4']['y']
    return [time,env1,env2]
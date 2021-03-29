from matplotlib import pyplot as plt
import json
import numpy as np
import math
from numpy.lib.function_base import gradient
import os

def ReadOriginalData(JsonFile,axis,joint):
    global ControlPoints
    ControlPoints = []

    global LocalGradients
    LocalGradients = []

    for i in range(0,16):
        ControlPoints.append([round(JsonFile["info"][joint][axis][i][0]),JsonFile["info"][joint][axis][i][1]])
        LocalGradients.append(JsonFile["info"][joint][axis][i][2])

def CaculateCubicFunction():
    # 计算各段三次函数的系数
    # y=ax^3+bx^2+cx+d
    global Coefficient
    Coefficient = []
    for i in range(1,len(ControlPoints)):
        x1 = ControlPoints[i-1][0]
        x2 = ControlPoints[i][0]
        y1 = ControlPoints[i-1][1]
        y2 = ControlPoints[i][1]
        k1 = LocalGradients[i-1]
        k2 = LocalGradients[i]

        if (k1 == k2) and (((y2-y1)/(x2-x1))==k1):
            a = 0
            b = 0
            c = k1
            d = y1-c*x1
        else:
            a = (2*(y1-y2)-(k1+k2)*(x1-x2))/(-pow(x1,3)+pow(x2,3)+3*pow(x1,2)*x2-3*pow(x2,2)*x1)
            b = (k1-k2-3*a*(x1-x2)*(x1+x2))/(2*(x1-x2))
            c = k1-3*a*pow(x1,2)-2*b*x1
            d = y1-a*pow(x1,3)-b*pow(x1,2)-c*x1

        Coefficient.append([a,b,c,d])    

def ReShow(AllPointNum): 
    # 计算误差值
    global KeyPoints
    KeyPoints = []
    for i in range(0,len(ControlPoints)-1):
        for j in range(ControlPoints[i][0],ControlPoints[i+1][0]):
            PresentY = Coefficient[i][0]*pow(j,3) + Coefficient[i][1]*pow(j,2) + Coefficient[i][2]*j + Coefficient[i][3]
            KeyPoints.append([j,PresentY])
    
    return KeyPoints

def ReDoInfo(JsonFile,axis,joint):
    ReadOriginalData(JsonFile,axis,joint)
    AllPointNum = JsonFile["info"][joint][axis][15][0]
    CaculateCubicFunction()
    return ReShow(AllPointNum)



def ReHermite(JsonFile,out_path):

    global outfile 
    outfile = {}

    outfile["name"] = JsonFile["name"]

    outfile["info"] = []

    for i in range(0,24):
        ALine = []
        for j in range(0,7):
            ALine.append(ReDoInfo(JsonFile,j,i))
        outfile["info"].append(ALine)
    json.dump(outfile,open(out_path,"w"))


if __name__ == "__main__":
    jsonfile = json.load(open("/home/shizhelun/PointNetMotion/PointNetMotion/experiments/example/20210304/stand-right_bow_step_Strike_07.json/origin.json"))
    ReHermite(jsonfile,"/home/shizhelun/PointNetMotion/PointNetMotion/experiments/example/20210304/stand-right_bow_step_Strike_07.json/out.json")

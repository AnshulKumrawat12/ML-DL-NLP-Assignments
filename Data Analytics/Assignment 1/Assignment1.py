# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 10:28:22 2021

@author: Anshul
"""

import csv 
import numpy as np
import pandas as pd
import math
import datetime
from math import sin, cos, tan, pi, atan
import matplotlib.pyplot as plt

#%%

# =============================================================================
# ---------------------------------Load Data-----------------------------------
# =============================================================================

# FILENAME = "D:\IISc\IISc 3rd Semester\DA\Assignment1\data_mars_opposition_updated.csv"
FILENAME = "data_mars_opposition_updated.csv"


data = pd.read_csv(FILENAME)
print(data.shape)


#%%

def degree_to_radian(deg):
    return deg * pi/180


def radian_to_degree(rad):
    return rad * 180/pi


def line_equation_with_two_points(p1, p2):
    m = (p2[1] - p1[1]) / (p2[0] - p1[0])
    c = p1[1]  - m * p1[0]  # y = mx + c
    return [m, c]

def line_equation_with_one_point(p1, angle):
    m = math.tan(degree_to_radian(angle))
    c = p1[1]  - m * p1[0]  # y = mx + c
    return [m, c]

def intersection_func(circle, line, angle):
    a = line[0]**2 + 1  
    b = 2 * (line[0]*line[1] - circle[0][0] - line[0]*circle[0][1]) 
    c = circle[0][0]**2 + circle[0][1]**2 + line[1]**2 - 2*circle[0][1]*line[1] - circle[1]**2


    if ((b)**2 - 4*a*c) < 0:
        x_1 = - b/(2*a)
        x_2 = - b/(2*a)
    
    else:
        x_1 = (- b + np.sqrt(((b)**2 - 4*a*c)))/(2*a) # -b + sqrt(b^2 - 4ac)
        x_2 = (- b - np.sqrt(((b)**2 - 4*a*c)))/(2*a) # -b - sqrt(b^2 - 4ac)
    
    
    y_1 = line[0] * x_1 + line[1]
    y_2 = line[0] * x_2 + line[1]
    
    
    if angle>0 and angle<= 90:
        if x_1 >= x_2:    
            return [x_1, y_1]
        else:
            return [x_2, y_2]
        
        
    elif angle>90 and angle<=180:
        if x_1 <= x_2:    
            return [x_1, y_1]
        else:
            return [x_2, y_2]
        
    elif angle>180 and angle<=270:
        if x_1 <= x_2:    
            return [x_1, y_1]
        else:
            return [x_2, y_2]
        
    else:
        if x_1 >= x_2:    
            return [x_1, y_1]
        else:
            return [x_2, y_2] 


#%%
Longitude = []
Days = []

# datetime(year, month, day, hour, minute, second)
Date_reference = datetime.datetime(data['Year'][0], data['Month'][0], data['Day'][0], data['Hour'][0], data['Minute'][0] )
Days.append(0)

for i in range(1,12):
    #DAYS
    a = datetime.datetime(data['Year'][i], data['Month'][i], data['Day'][i], data['Hour'][i], data['Minute'][i] ) 
    diff = a - Date_reference  
    days = diff.total_seconds()/(60*60*24)
    Days.append(days)
    
for i in range(0,12):
    #lONGITUDE
    b = data['ZodiacIndex'][i]*30 + data['Degree'][i] + data['Minute.1'][i]/60 + data['Second'][i]/3600
    Longitude.append(b)

oppositions = np.column_stack((Days, Longitude))


#%%

def MarsEquantModel(c,r,e1,e2,z,s,oppositions):
    #Initialize points of Sun, center, equant
    sun = [0,0]
    center = [1*cos(degree_to_radian(c)), 1*sin(degree_to_radian(c))] #(r*cos(), r*sin())
    equant = [e1*cos(degree_to_radian(e2 + z)), e1*sin(degree_to_radian(e2 + z))] #(e1*cos(e2+z), e1*sin(e2+z))

    #Center Points
    circle = [center, r]
    errors = []
    max_error = 0
    
    
    x_point = []
    y_point = []
    
    sun_x = []
    sun_y = []
   
    
    for i in range(0,12):
        
        angle = (oppositions[i][0]*s + z) % 360 
        
        #solid line
        solid_line = line_equation_with_one_point(sun, angle = oppositions[i][1])
  
        #dotted Line from equant
        dotted_line = line_equation_with_one_point(equant, angle = angle)
        
        #Intersection point on circle from equant
        intersection_of_equant = intersection_func(circle, dotted_line, angle)
        x_point.append(intersection_of_equant[0])
        y_point.append(intersection_of_equant[1])
        
        #Intersection of solid line of sun from circle
        intersection_of_sun= intersection_func(circle, solid_line, oppositions[i][1])
        sun_x.append(intersection_of_sun[0])
        sun_y.append(intersection_of_sun[1])
        
        #Dotted line from sun
        dottedline_from_sun = line_equation_with_two_points(p1=sun, p2=intersection_of_equant)
        
        #Angle Difference 
        temp = atan(np.abs((solid_line[0] - dottedline_from_sun[0])/ (1 + solid_line[0]*dottedline_from_sun[0])))
    
        temp = radian_to_degree(temp)
        errors.append(temp)
        
        if temp > max_error:
            max_error = temp  
    
    
# =============================================================================
#   -------------------------Graph Plot --------------------------------------- 
#         figure, axes = plt.subplots()
#         for i in range(12):
#         x_values = [equant[0], x_point[i]]
#         y_values = [equant[1], y_point[i]]
#         
#         x_values1 = [sun[0], sun_x[i]]
#         y_values1 = [sun[1], sun_y[i]]
#         
#         
#         plt.plot(x_values, y_values, 'orange', linestyle='--', linewidth=0.5)
#         plt.plot(x_values1, y_values1, 'red', linewidth=0.5)
#     
#     plt.scatter(x_point,y_point, color='yellow')
#     plt.scatter(sun_x, sun_y, color='lightcoral')
#     
#     plt.arrow(0, 0, 6,0 ,
#           head_width = 0.1,
#           width = 0.001,
#           ec ='black')
#     plt.plot(sun[0],sun[1],label = 'Sun(solid)', color = 'lightcoral')
#     plt.plot(center[0], center[1], color = 'green')
#     plt.plot(equant[0], equant[1],label ='Equant(dotted)', color = 'orange')
#     axes.set_aspect(1)
#     draw_circle = plt.Circle((center[0], center[1]), r,fill=False)
#     axes.add_artist(draw_circle)
#     plt.legend(loc = 'upper right')
#     plt.show()
# =============================================================================

    return np.array(errors), max_error

#%%

#Question 02
def bestOrbitInnerParams(r,s,oppositions):
    minimum = 200
    C = 152.9021
    c=[]
    c.append(C)
    for i in range(100):
        c.append(c[-1]+0.000001)
    
    E1 = 1.544555
    e1 = []
    e1.append(E1)
    for i in range(100):
        e1.append(e1[-1]+0.00000001)
    
    E2 = 93.1975
    e2 = []
    e2.append(E2)
    for i in range(100):
        e2.append(e2[-1]+0.000001)
    
    Z = 55.74278
    z = []
    z.append(C)
    for i in range(100):
        z.append(z[-1]+0.0000001)
        
    for i in c:
        for j in e1:
            for k in e2:
                for l in z:
                    errors, max_error = MarsEquantModel(i,r,j,k,l,s,oppositions)
                    
                    if minimum > max_error:
                        minimum = max_error
                        m_c = i
                        m_e1 = j
                        m_e2 = k
                        m_z = l
                    
    
    errors, max_error = MarsEquantModel( m_c,r, m_e1, m_e2, m_z,s, oppositions)
    return m_c, m_e1, m_e2, m_z, errors, max_error


#%%
#Question 03

def bestS(r,oppositions):
    minimum = 200
    S = 0.524087
    s = []
    s.append(S)
    for i in range(100):
        s.append(s[-1] + 0.00000001)

    for i in s:
        c, e1 ,e2, z, errors, max_error = bestOrbitInnerParams(r,i,oppositions)

        if minimum > max_error:
            minimum = max_error
            m_s = i
            m_c = c
            m_e1 = e1
            m_e2 = e2
            m_z = z
       
    errors, maxError = MarsEquantModel(m_c, r, m_e1, m_e2, m_z, m_s, oppositions)
    return m_s, errors, maxError

    
#%%
#Question 04

def bestR(s,oppositions):
    minimum = 200
    R = 8.317
    r = []
    r.append(R)
    for i in range(100):
        r.append(r[-1] + 0.00001)

    for i in r:
        c, e1 ,e2, z, errors, max_error = bestOrbitInnerParams(i,s,oppositions)

        if minimum > max_error:
            minimum = max_error
            m_r = i
            m_c = c
            m_e1 = e1
            m_e2 = e2
            m_z = z
       
    errors, maxError = MarsEquantModel(m_c, m_r, m_e1, m_e2, m_z, s, oppositions)
    return m_r, errors, maxError


#%%

#Question 05
def bestMarsOrbitParams(oppositions):
    minimum = 200
    R = 8.317
    r = []
    r.append(R)
    for i in range(100):
        r.append(r[-1] + 0.00001)
        
    S = 0.524087
    s = []
    s.append(S)
    for i in range(100):
        s.append(s[-1] + 0.00000001)

    
    for i in r:
        for j in s:        
            c, e1 ,e2, z, errors, max_error = bestOrbitInnerParams(i,j,oppositions)
    
            if minimum > max_error:
                minimum = max_error
                m_r = i
                m_s = j
                m_c = c
                m_e1 = e1
                m_e2 = e2
                m_z = z
       
    errors, maxError = MarsEquantModel(m_c, m_r, m_e1, m_e2, m_z, m_s, oppositions)
    return m_r,m_s, m_c, m_e1, m_e2, m_z, errors, maxError


#%%

def main():
    
    #Function calling 1
    c = 152.902143
    r = 8.317030
    e1 = 1.54455535
    e2 = 93.1975230
    z =  55.7427834
    s = 0.52408795
    
    # MarsEquantModel(c,r,e1,e2,z,s,oppositions)
    errors1, max_error1 = MarsEquantModel(c, r, e1, e2, z, s, oppositions)
    print("-------- Question 01 ---------")
    
    print("c: {}, r : {}, e1: {}, e2: {}, z: {}, s: {}".format(c,r,e1,e2,z,s))
    print("Error :", errors1)
    print(" ------### Max - Error :{} ### ------ ".format(max_error1))


    #Function Calling 2
    # r = 8.31703
    # s = 0.52408795
    # m_c, m_e1, m_e2, m_z, errors2, maxError2 =  bestOrbitInnerParams(r,s,oppositions)
    
    # print("-------- Question 02 ---------")
    # #print("Error :", errors2)
    # print("Max - Error : ", maxError2)

        
    ###--- Function Calling 3
    # s,errors,maxError3 = bestS(r,oppositions)
    
    # print("-------- Question 03 ---------")
    # print("s = ", s)
    # #print("Error :", errors)
    # print("Max - Error : ", maxError3)



    ###--- Function Calling 4
    # r,errors,maxError4 = bestR(s,oppositions)
    
    # print("r = ", r)
    # print("-------- Question 04 ---------")
    # #print("Error :", errors)
    # print("Max - Error : ", maxError4)
    
    
    
    ###--- Function Calling 5
    # r,s,c,e1,e2,z,errors,maxError5 = bestMarsOrbitParams(oppositions)
    
    # print("-------- Question 05 ---------")
    # #print("Error :", errors)
    # print("Max - Error : ", maxError5)

#%%

if __name__== "__main__":
    main()
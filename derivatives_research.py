import numpy as np
import matplotlib.pyplot as plt
from sympy import *
import math

x = symbols('x') #intercept
y = symbols('y') #slope

data = [[5, 9, 6], [7, 2, 4]] #not work with this data set
#data = [[0.5, 2.3, 2.9],[1.4, 1.9, 3.2]]

learning_rate = 0.01
intercept = 0
slope = 1
observed = 0
predicted = 0
sumed2_resid = 0
dx_int = 0
dx_slope = 0

for iterations in range(int(len(data[0]))):
    weight = data[0][iterations]
    observed = data[1][iterations]
    predicted = (weight*y) + x
    residual = observed - predicted
    sumed2_resid = sumed2_resid + pow(residual, 2)
    dx_int = dx_int + diff(pow(residual, 2), x) #possibly error with pulling derivatives, check variable continuity
    dx_slope = dx_slope + diff(pow(residual, 2), y)

ix = dx_int.subs({x:intercept, y: slope})
sx = dx_slope.subs({x: intercept, y: slope})

print(f'ix: {ix}, sx: {sx}') #seems correct up until here(?)

intStep_size = ix * learning_rate
slopeStep_size = sx * learning_rate 

old_int = intercept
old_slope = slope
new_int = old_int - intStep_size
new_slope = old_slope - slopeStep_size

for times in range(1000):
    ix = dx_int.subs({x:new_int, y: new_slope})
    sx = dx_slope.subs({x: new_int, y: new_slope})

    intStep_size = ix * learning_rate
    slopeStep_size = sx * learning_rate

    old_int = new_int
    old_slope = new_slope
    new_int = old_int - intStep_size
    new_slope = old_slope - slopeStep_size
    
    if(abs(intStep_size) <= 0.00001 or abs(slopeStep_size) <= 0.00001):
        print('breaking ' + str(times) + ': ' + str(new_int) + '|' + str(new_slope))
        break

result_int = round(new_int, 2)
result_slope = round(new_slope, 2)
print(f'y = {result_slope}x + {result_int}')

#graphing stuff
##line_x = [0] * (int(len(data[0])) +2)
##line_y = [0] * (int(len(data[0]))+2)
##x_coord = data[0]
##y_coord = data[1]
##
##largest = data[0][0]
##for value in range(int(len(data[0]))):
##    if(data[0][value] > largest):
##        largest = data[0][value]
##for value in range(int(len(data[1]))):
##     if(data[1][value] > largest):
##        largest = data[1][value]
##
##for points in range (int(len(data[0]))):
##    line_x[points] = data[0][points]
##    line_y[points] = (line_x[points]*result_slope) + result_int
##line_x[-2] = 0
##line_x[-1] = (largest + 1)
##line_y[-2] = result_int
##line_y[-1] = ((largest + 1) * result_slope) + result_int
##
##newX_coord = line_x[0:3]
##newY_coord = line_y[0:3]
##
##for element in range(len(x_coord)):
##    coordinates = (x_coord[element-1], y_coord[element-1])
##    new_coordinates = (line_x[element-1], round(line_y[element-1], 2))
##    plt.text(x_coord[element-1]+0.05, y_coord[element-1]+0.05, coordinates)
##    plt.text(newX_coord[element-1] - 0.25, newY_coord[element-1] - 0.25, new_coordinates)
##plt.xlabel('Weight (Ibs)')
##plt.ylabel('Heihgt (Inch)')
##
##plt.plot([0, (largest+1)], [0, (largest+1)])
##plt.plot(data[0], data[1], 'bo')
##plt.plot(line_x, line_y)
##plt.plot(line_x[0:3], line_y[0:3], 'ro')
##plt.show()

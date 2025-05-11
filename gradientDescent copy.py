import numpy as np
import matplotlib.pyplot as plt
from sympy import *
import math

#initialize equation's variables
x = symbols('x') #intercept
y = symbols('y') #slope

data = [[20, 50, 70], [1, 2, 6]]


#need to adjust/tune algorithm for LR, runtime, thesholds, etc

#find the largest and smallest values that are given
largest = data[0][0]
smallest = data[0][0]

for axis_x in range(int(len(data[0]))):
    if(data[0][axis_x] > largest):
        largest = data[0][axis_x]
    if(data[0][axis_x] < smallest):
        smallest = data[0][axis_x]
for axis_y in range(int(len(data[1]))):
    if(data[1][axis_y] > largest):
        largest = data[1][axis_x]
    if(data[1][axis_y] < smallest):
        smallest = data[1][axis_y]

#set up variables
learning_rate = 0.0001
intercept = 0
slope = 1
observed = 0
predicted = 0
sumed2_resid = 0
dx_int = 0
dx_slope = 0

#create cost function and take dx
for iterations in range(int(len(data[0]))):
    weight = data[0][iterations]
    observed = data[1][iterations]
    predicted = (weight*y) + x
    residual = observed - predicted
    sumed2_resid = sumed2_resid + pow(residual, 2)
    dx_int = dx_int + diff(pow(residual, 2), x)
    dx_slope = dx_slope + diff(pow(residual, 2), y)

ix = dx_int.subs({x:intercept, y: slope})
sx = dx_slope.subs({x: intercept, y: slope})

#print(f'dx_int: {dx_int}, dx_slope: {dx_slope}')
#print(f'ix: {ix}, sx: {sx}')

intStep_size = ix * learning_rate
slopeStep_size = sx * learning_rate

old_int = intercept
old_slope = slope
new_int = old_int - intStep_size
new_slope = old_slope - slopeStep_size


#descend gradient
print(f'calculating... ')
for times in range(100000):    
    ix = dx_int.subs({x:new_int, y: new_slope})
    sx = dx_slope.subs({x: new_int, y: new_slope})
        
    intStep_size = ix * learning_rate
    slopeStep_size = sx * learning_rate
    
    old_int = new_int
    old_slope = new_slope
    new_int = old_int - intStep_size
    new_slope = old_slope - slopeStep_size
    
#break cases
    if(intercept != 0 and slope != 0):
        if(abs(intStep_size) <= 0.00000001 and abs(slopeStep_size) <= 0.00000001):
            print('breaking ' + str(times) + ': ' + str(new_int) + '|' + str(new_slope) + '(0)') #case 1
            break
    elif(intercept == 0 and slope != 0):
        if(abs(slopeStep_size) <= 0.00000001):
            print('breaking ' + str(times) + ': ' + str(new_int) + '|' + str(new_slope) + '(1)') #case 2
            break
    elif(intercept != 0 and slope == 0):
        if(abs(intStep_size) <= 0.00000001):
            print('breaking ' + str(times) + ': ' + str(new_int) + '|' + str(new_slope) + '(2)') #case 3
            break
    else:
        print('completed loop without breaking')

#print out results
result_int = round(new_int, 2)
result_slope = round(new_slope, 2)
print(f'y = {result_slope}x + {result_int}')

###create coordinates for lines
##direct = []
##for index in range(int(len(data[0]))): 
##    direct.append(data[0][index])
##direct.append(largest + (largest * 0.5))
##direct.append(smallest - (smallest * 0.5))
##
##regression_x = []
##for index in range(int(len(data[0]))):
##    regression_x.append(data[0][index])
##regression_x.append(largest + (largest * 0.5))
##regression_x.append(smallest - (abs(smallest) * 0.5))
##
##regression_y = ([0] * (int(len(data[0])) + 2))
##for index in range(int(len(data[0]))):
##    regression_y[index] = (regression_x[index] * result_slope) + result_int
##regression_y[-2] = (regression_x[-2] * result_slope) + result_int
##regression_y[-1] = (regression_x[-1] * result_slope) + result_int
##
##plt.plot([smallest - (abs(smallest) * 0.5), 0, largest + (largest * 0.5)],[0, 0, 0], color = 'black') #x_axis
##plt.plot([0, 0, 0], [smallest - (abs(smallest) * 0.5), 0, largest + (largest * 0.5)], color = 'black') #y_axis
##
###make the axis arorws
##plt.arrow(smallest - (abs(smallest) * 0.5), 0, 0.5, 0.5)
##plt.arrow(smallest - (abs(smallest) * 0.5), 0, 0.5, -0.5) #fix the arrow width depending on scale
##plt.arrow(largest + (largest * 0.5), 0, -0.5, 0.5)
##plt.arrow(largest + (largest * 0.5), 0, -0.5, -0.5)
##plt.arrow(0, largest + (largest * 0.5), 0.5, -0.5)
##plt.arrow(0, largest + (largest * 0.5), -0.5, -0.5)
##plt.arrow(0, smallest - (abs(smallest) * 0.5), 0.5, 0.5)
##plt.arrow(0, smallest - (abs(smallest) * 0.5), -0.5, 0.5)
##
##
##plt.plot(direct, direct, color = 'orange') #y=x
##plt.text(smallest, smallest + (largest * 0.6), 'y = x')
##
##plt.plot(regression_x, regression_y, color = 'c') #line of best fit
##plt.plot(regression_x[0:int(len(data[0]))], regression_y[0:int(len(data[0]))], 'bo')
##plt.text(smallest, largest + (largest * 0.5), f'y = {result_slope}x + {result_int}')
##
##plt.plot(data[0], data[1], 'ro') #data points
##for coordinate in range(int(len(data[0]))):
##    plt.text(data[0][coordinate] - (largest * 0.5), data[1][coordinate] - (largest * 0.5), f'({data[0][coordinate]}, {data[1][coordinate]})')
##plt.show()


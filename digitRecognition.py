import numpy as np
import cv2 as cv
import random

def sig(x):
    return 1/(1 + np.exp(-x))

#getting data from image in greyscale
image_width = 28
image_length = 28
pixels = np.array([[0]*image_width]*image_length)

picture = input("Paste the path of the image you would like to use:")
img = cv.imread(picture, 0)

for length_pixels in range(28):
  for width_pixels in range(28):
    pixels[length_pixels][width_pixels] = img[length_pixels][width_pixels]/255

print(pixels)      

#create the weights for each connection
weights = np.array([[0]*image_width]*image_length)
for weights_length in range(28):
  for weights_width in range(28):
    weights[weights_length][weights_width] = round(random.uniform(10, 99), 3)
#print(weights/100)


#combine the wieghts to get the output value
combined = np.array([[0]*image_width]*image_length)
for combined_length in range(28):
  for combined_width in range(28):
    combined[combined_length][combined_width] = pixels[combined_length][combined_width] * weights[combined_length][combined_width]
combined = combined/100
#print(combined)

#stick lower edge
edge_nine = 0
for vertical_edgeNine in range(12, 27):
    for horizontal_edgeNine in range(17, 25):
        edge_nine = edge_nine + combined[vertical_edgeNine][horizontal_edgeNine]
#straight part of loop
edge_eight = 0
for vertical_edgeEight in range(4, 15):
    for horizontal_edgeEight in range(17, 25):
        edge_eight = edge_eight + combined[vertical_edgeEight][horizontal_edgeEight]

#loop top connector
edge_one = 0
for vertical_edgeOne in range(2, 9):
    for horizontal_edgeOne in range(16, 22):
        edge_one = edge_one + combined[vertical_edgeOne][horizontal_edgeOne]

#loop bottom connector
edge_seven = 0
for vertical_edgeSeven in range(11, 17):
    for horizontal_edgeSeven in range(15, 20):
        edge_seven = edge_seven + combined[vertical_edgeSeven][horizontal_edgeSeven]

#top straight edge
edge_two = 0
for vertical_edgeTwo in range(2, 7):
    for horizontal_edgeTwo in range(9, 20):
        edge_two = edge_two + combined[vertical_edgeTwo][horizontal_edgeTwo]

#bottom straight edge
edge_six = 0
for vertical_edgeSix in range(12, 18):
    for horizontal_edgeSix in range(8, 18):
        edge_six = edge_six + combined[vertical_edgeSix][horizontal_edgeSix]

#left-top connector edge
edge_three = 0
for vertical_edgeThree in range(3, 10):
    for horizontal_edgeThree in range(5, 11):
        edge_three = edge_three + combined[vertical_edgeThree][horizontal_edgeThree]

#left-bottom connector edge
edge_five = 0
for vertical_edgeFive in range(12, 17): #enter values
    for horizontal_edgeFive in range(5, 10):
        edge_five = edge_five + combined[vertical_edgeFive][horizontal_edgeFive]

#outermost edge
edge_four = 0
for vertical_edgeFour in range(7, 15): #enter values
    for horizontal_edgeFour in range(3, 8):
        edge_four = edge_four + combined[vertical_edgeFour][horizontal_edgeFour]

biases = [0]*9

for biase_place in range(9):
    biases[biase_place] = random.uniform(5, 10)
#print(biases)

edgeOneOutput = round(sig(edge_one-biases[0]), 4)
edgeTwoOutput = round(sig(edge_two-biases[1]), 4)
edgeThreeOutput = round(sig(edge_three-biases[2]), 4)
edgeFourOutput = round(sig(edge_four-biases[3]), 4)
edgeFiveOutput = round(sig(edge_five-biases[4]), 4)
edgeSixOutput = round(sig(edge_six-biases[5]), 4)
edgeSevenOutput = round(sig(edge_seven-biases[6]), 4)
edgeEightOutput = round(sig(edge_eight-biases[7]), 4)
edgeNineOutput = round(sig(edge_nine-biases[8]), 4)

#stick shape
stick = sig(edgeEightOutput + edgeNineOutput)
#print(stick)
if(stick >= 0.65):
    stick = 1
else:
    stick = 0
#loop shape
loop = sig(edgeOneOutput + edgeTwoOutput + edgeThreeOutput + edgeFourOutput + edgeFiveOutput + edgeSixOutput + edgeSevenOutput)
if(loop >= 0.8):
    loop = 1
else:
    loop = 0

if(loop == 1 and stick == 1):
    print("It's a nine")
else:
    print("no")

#put the wieghts and biases into a function and call it once
#use gradient descent to adjust wieghts and biases

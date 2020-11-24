## MAIN ###
from hough import hough
from orientation_estimate import *


flag = Flag(input("Enter path, e.g: C:/marian/imagenes"),input("Enter image name, e.g: flag.png"))

num_colors = flag.colors()
percentage = flag.percentage()
orientation = flag.orientation()

print("The number of colors is: ",num_colors)
print("The percentage of each color is: ", percentage)
print("The flag is: ",orientation)
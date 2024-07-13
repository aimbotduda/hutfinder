import re

# i have the following outputs in a file
# ([array([333312, 601088]), array([333312, 600864]), array([333168, 600896])], 2061515.2163536)
# ([array([333168, 600896]), array([333312, 601088]), array([333152, 601120])], 2061661.516959644)

# clean the output file, remove the array formates, keep the coordinates and the distance

import re
import math

# Function to clean the line and calculate the distance sum
def clean_line(line):
    # Use regex to find all coordinates including negative ones and handle spaces
    coordinates = re.findall(r'\[\s*(-?\d+),\s*(-?\d+)\s*\]', line)
    # Use regex to find the distance
    distance = re.findall(r'\d+\.\d+', line)[-1]
    
    # Sort the coordinates to ensure the same group is always in the same order
    coordinates = sorted(coordinates, key=lambda x: (int(x[0]), int(x[1])))
    
    # Calculate the sum of distances between each pair of coordinates
    def euclidean_distance(coord1, coord2):
        return math.sqrt((int(coord1[0]) - int(coord2[0]))**2 + (int(coord1[1]) - int(coord2[1]))**2)
    
    distance_sum = (
        euclidean_distance(coordinates[0], coordinates[1]) +
        euclidean_distance(coordinates[1], coordinates[2]) +
        euclidean_distance(coordinates[2], coordinates[0])
    )
    
    # Prepare the cleaned line with the required format
    cleaned_coordinates = ' '.join([f'({x}, {y})' for x, y in coordinates])
    cleaned_line = f'{cleaned_coordinates} {distance} {distance_sum:.2f}'
    
    return cleaned_line, distance_sum

# Use a list to store unique cleaned lines and their distance sums
unique_lines = []

# Read the input file
with open('outputs/output4Mon.txt', 'r') as infile:
    for line in infile:
        cleaned_line, distance_sum = clean_line(line)
        if cleaned_line not in (line for line, _ in unique_lines):
            unique_lines.append((cleaned_line, distance_sum))

# Sort the unique lines by distance_sum
unique_lines.sort(key=lambda x: x[1])

# Write the sorted lines to the output file
with open('cleaned_output.txt', 'w') as outfile:
    for cleaned_line, _ in unique_lines:
        outfile.write(cleaned_line + '\n')

print("Data cleaned, duplicates removed, sorted by distance_sum, and written to cleaned_output.txt")




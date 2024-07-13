import numpy as np
from scipy.spatial import cKDTree

def check_group(coords):
    for i in range(len(coords)):
        for j in range(i+1, len(coords)):
            dx = coords[i][0] - coords[j][0]
            dz = coords[i][1] - coords[j][1]
            dist = (dx*dx + dz*dz) ** 0.5
            if dist > 200:
                return False
    return True

#function to find center coordinates of a group
def find_center(coords):
    x = 0
    z = 0
    for i in range(len(coords)):
        x += coords[i][0]
        z += coords[i][1]
    x /= len(coords)
    z /= len(coords)
    print (x, z)
    return (x, z)

# Parse the (x, z) coordinates of each place from the text file
print("Reading and parsing file of all monuments")
num_lines = sum(1 for _ in open('outputs/allhuts.txt'))
processedLines = 0
places = []
with open("outputs/allhuts.txt") as f:
    for line in f:
        if "->(" in line:
            # show progress 
            if len(places) % 5000000 == 0:
                percentage = (processedLines / num_lines) * 100
                percentage = round(percentage, 2)
                print(f"{percentage}% read")
            coords = line.split("->(")[1].split(")reg")[0]
            x, z = map(int, coords.split(","))
            places.append((x, z))
        processedLines += 1
places = np.array(places)
print("Parsed file")
print(f"Found {len(places)} places")



# Build a kd-tree from the (x, z) coordinates of the places
print("Building tree... this may take a while")
tree = cKDTree(places)
print("Built tree")


def find_groups_3():
    groups = []
    found = 0
    totalPlaces = len(places)
    # Search for groups of three places that are within 160 blocks of each other
    for i in range(len(places)):
        # show progress
        if i % 100000 == 0:
            percentage = (i / totalPlaces) * 100
            percentage = round(percentage, 2)
            print(f"{percentage}% searched - Found {found} groups")
        # Find all places within 160 blocks of the current place
        neighbors = tree.query_ball_point(places[i], r=200)
        # Check if any group of three places is within 160 blocks of each other
        for j in range(len(neighbors)):
            if neighbors[j] != i:
                for k in range(j+1, len(neighbors)):
                    if neighbors[k] != i:
                        if check_group([places[i], places[neighbors[j]], places[neighbors[k]]]):
                            dist = ((places[i][0] ** 2 + places[i][1] ** 2) ** 0.5 +
                                    (places[neighbors[j]][0] ** 2 + places[neighbors[j]][1] ** 2) ** 0.5 +
                                    (places[neighbors[k]][0] ** 2 + places[neighbors[k]][1] ** 2) ** 0.5)
                            groups.append(([places[i], places[neighbors[j]], places[neighbors[k]]], dist))
                            found += 1

    groups.sort(key=lambda x: x[1])
    with open("output3Mon.txt", "a") as f:
        for group in groups:
            f.write(f"{group}")
            f.write("\n")

def find_groups_4():
    groups = []
    found = 0
    totalPlaces = len(places)
    # Search for groups of four places that are within 160 blocks of each other
    for i in range(len(places)):
        # show progress
        if i % 100000 == 0:
            percentage = (i / totalPlaces) * 100
            percentage = round(percentage, 2)
            print(f"{percentage}% searched - Found {found} groups")
        # Find all places within 160 blocks of the current place
        neighbors = tree.query_ball_point(places[i], r=200)
        # Check if any group of four places is within 160 blocks of each other
        for j in range(len(neighbors)):
            if neighbors[j] != i:
                for k in range(j+1, len(neighbors)):
                    if neighbors[k] != i:
                        for l in range(k+1, len(neighbors)):
                            if neighbors[l] != i:
                                if check_group([places[i], places[neighbors[j]], places[neighbors[k]], places[neighbors[l]]]):
                                    dist = ((places[i][0] ** 2 + places[i][1] ** 2) ** 0.5 +
                                            (places[neighbors[j]][0] ** 2 + places[neighbors[j]][1] ** 2) ** 0.5 +
                                            (places[neighbors[k]][0] ** 2 + places[neighbors[k]][1] ** 2) ** 0.5 +
                                            (places[neighbors[l]][0] ** 2 + places[neighbors[l]][1] ** 2) ** 0.5)
                                    groups.append(([places[i], places[neighbors[j]], places[neighbors[k]], places[neighbors[l]]], dist))
                                    found += 1

    groups.sort(key=lambda x: x[1])
    with open("output4Mon.txt", "a") as f:
        for group in groups:
            f.write(f"{group}")
            f.write("\n")


find_groups_3()
find_groups_4()
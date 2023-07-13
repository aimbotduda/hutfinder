from scipy.spatial import cKDTree

def check_group(coords):
    for i in range(len(coords)):
        for j in range(i+1, len(coords)):
            dx = coords[i][0] - coords[j][0]
            dz = coords[i][1] - coords[j][1]
            dist = (dx*dx + dz*dz) ** 0.5
            if dist >= 256:
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
with open("huts.txt") as f:
    lines = f.readlines()

places = []
for line in lines[1:]:
    parts = line.strip().split(";")
    x = int(parts[2])
    z = int(parts[3])
    places.append((x, z))

print(f"Found {len(places)} places")
# Build a kd-tree from the (x, z) coordinates of the places
tree = cKDTree(places)
print("Built tree")

def find_groups_4():
    # Search for groups of four places that are within 256 blocks of each other
    for i in range(len(places)):
        # Find all places within 256 blocks of the current place
        neighbors = tree.query_ball_point(places[i], r=256)

        # Check if any group of four places is within 256 blocks of each other
        for j in range(len(neighbors)):
            for k in range(j+1, len(neighbors)):
                for l in range(k+1, len(neighbors)):
                    if check_group([places[i], places[neighbors[j]], places[neighbors[k]], places[neighbors[l]]]):
                        print(f"Group found: {places[i]}, {places[neighbors[j]]}, {places[neighbors[k]]}, {places[neighbors[l]]}")
                        with open("output4.txt", "a") as f:
                            f.write(f"{places[i]}, {places[neighbors[j]]}, {places[neighbors[k]]}, {places[neighbors[l]]}")

def find_groups_3():
    groups = []
    # Search for groups of three places that are within 256 blocks of each other
    for i in range(len(places)):
        # Find all places within 256 blocks of the current place
        neighbors = tree.query_ball_point(places[i], r=256)
        # Check if any group of three places is within 256 blocks of each other
        for j in range(len(neighbors)):
            for k in range(j+1, len(neighbors)):
                if neighbors[j] != i and neighbors[k] != i and check_group([places[i], places[neighbors[j]], places[neighbors[k]]]):
                    dist = ((places[i][0] ** 2 + places[i][1] ** 2) ** 0.5 +
                            (places[neighbors[j]][0] ** 2 + places[neighbors[j]][1] ** 2) ** 0.5 +
                            (places[neighbors[k]][0] ** 2 + places[neighbors[k]][1] ** 2) ** 0.5)
                    groups.append(([places[i], places[neighbors[j]], places[neighbors[k]]], dist))
    groups.sort(key=lambda x: x[1])
    with open("output3.txt", "a") as f:
        for group in groups:
            f.write(f"{group}")
            f.write("\n")

def find_groups_2():
    groups = []
    # Search for groups of two places that are within 256 blocks of each other
    for i in range(len(places)):
        # Find all places within 256 blocks of the current place
        neighbors = tree.query_ball_point(places[i], r=256)
        # Check if any group of two places is within 256 blocks of each other
        for j in range(len(neighbors)):
            if neighbors[j] != i and check_group([places[i], places[neighbors[j]]]):
                dist = ((places[i][0] ** 2 + places[i][1] ** 2) ** 0.5 +
                        (places[neighbors[j]][0] ** 2 + places[neighbors[j]][1] ** 2) ** 0.5)
                groups.append(([places[i], places[neighbors[j]]], dist))
    groups.sort(key=lambda x: x[1])
    with open("output2.txt", "a") as f:
        for group in groups:
            f.write(f"{group}")
            f.write("\n")

#delete the output files
# open('output2.txt', 'w').close()
# open('output3.txt', 'w').close()
# open('output4.txt', 'w').close()
# find_groups_3()
# find_groups_4()
# find_groups_2()

def delete():
    #delete every second line of outputfile2
    with open("output2.txt", "r") as f:
        lines = f.readlines()
    with open("output2.txt", "w") as f:
        for i in range(0, len(lines), 2):
            f.write(lines[i])
            
# delete()

find_center([(333152, 601120), (333168, 600896), (333312, 600864), (333312, 601088)])
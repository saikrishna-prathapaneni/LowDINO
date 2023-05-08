import os

dir = 'val/'
actual='val/val_2/'
filename = "map_clsloc.csv"
classes_img = "GT.txt"
dictionary = {}
ground_truth =[]

with open(classes_img, "r") as file:
    for line in file:
        value = line.strip()
        ground_truth.append(int(value))
        
# print(len(ground_truth))
with open(filename, "r") as file:
    for line in file:
        print(line)
        key, value = line.strip().split(",")
        dictionary[key] = int(value)

for key in list(dictionary.keys()):
    print(actual+key)
    os.makedirs(actual+key)

for key in list(dictionary.keys()):
    for file in os.listdir(dir):
        if file =='val_2':
            continue
        #if list(dictionary.keys())[list(dictionary.values()).index(int(file[-10:-5]))] == key:
            #print("file name ", file, dictionary[int(file[-10:-5])], key )
        os.replace(dir+file,actual+str( list(dictionary.keys())[list(dictionary.values()).index(ground_truth[int(file[-10:-5])-1])])+'/'+file)

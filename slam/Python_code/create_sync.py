import sys
import os
import numpy


def read_file_list(filename):
    """
    Description
    Reads rgb.txt and depth.txt file with timestamps for each and matches each one
    with the closest one  
    
    File format:
    The file format is "stamp d1 d2 d3 ...", where stamp denotes the time stamp (to be matched)
    and "d1 d2 d3.." is arbitary data 
    """
    
    file = open(filename)
    data = file.read()
    lines = data.replace(","," ").replace("\t"," ").split("\n") 
    list = [[v.strip() for v in line.split(" ") if v.strip()!=""] for line in lines if len(line)>0 and line[0]!="#"]
    list = [(float(l[0]),l[1:]) for l in list if len(l)>1]
    return dict(list)

path=os.getcwd() + "/rgbd_dataset_freiburg2_pioneer_slam/"

first_list = read_file_list(path + 'rgb.txt')
aux1=list(first_list.keys())
del first_list

second_list = read_file_list(path + 'depth.txt')
aux2=list(second_list.keys())
del second_list

offset=0
max_difference=0.02
potential_matches = [(abs(a - (b + offset)),a, b) 
                         for a in aux1 
                         for b in aux2 
                         if abs(a - (b + offset)) < max_difference]
potential_matches.sort()

matches=[]
for diff, a, b in potential_matches:
        if a in aux1 and b in aux2:
            aux1.remove(a)
            aux2.remove(b)
            matches.append((a, b))

matches.sort()

output_file="sync.txt"
f=open(output_file,'w')
for i in range(len(matches)):
    aux3=str(matches[i][0])
    aux4=str(matches[i][1])

    while(1):
        if(len(aux3) <17):
            aux3= aux3 + '0'
        else:
            break
    while(1):
        if(len(aux4) <17):
            aux4= aux4 + '0'
        else:
            break    
        
    
    f.write(aux3+ " " +aux4 + "\n")
f.close()
print("Matches are written into filename " + output_file)




import pandas as pd
import os
import csv  
import shutil
import time

from argparse import ArgumentParser
from pathvalidate.argparse import validate_filename_arg, validate_filepath_arg

parser = ArgumentParser()
parser.add_argument("--filepath", type=validate_filepath_arg, help='write filepath to the *.ulg')
parser.add_argument("--filename", type=validate_filename_arg, help='write filename *.ulg')
parser.add_argument("--output",type=validate_filename_arg, help='output csv file name')
#parser.add_help()
options = parser.parse_args()

if options.filename:
    print("filename: {}".format(options.filename))

if options.filepath:
    print("filepath: {}".format(options.filepath))

os.mkdir("temp1")
os.system("ulog2csv --messages 'vehicle_imu_status' --output temp1 "+ options.filename)

header = ['#timestamp [ns]','w_RS_S_x [rad s^-1]','w_RS_S_y [rad s^-1]','w_RS_S_z [rad s^-1]','a_RS_S_x [m s^-2]','a_RS_S_y [m s^-2]','a_RS_S_z [m s^-2]']

if options.output:
    f=open(options.output + '.csv', 'w')
else:
    f=open('csv_file.csv', 'w')


g = open('temp1/log_3_2021-7-27-17-05-06_vehicle_imu_status_0.csv')
    
# create the csv writer
writer = csv.writer(f)

# create the csv reader
csv_reader = csv.reader(g)

# write a row to the csv file
writer.writerow(header)

# skip header row
if csv_reader:
    next(csv_reader)

for line in csv_reader:
    aux=[line[0], line[18], line[19],line[20], line[15],line[16],line[17]]
    writer.writerow(aux)
        

shutil.rmtree("temp1")
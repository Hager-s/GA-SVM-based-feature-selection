import pandas as pd
import numpy as np


try:
    f=open("GSE10810_series_matrix.txt")
except IOError:
    print("File myfile.fa does not exist!!")

Y = []
for line in f:
    line = line.rstrip()
    if line[29:55] == 'tumor (t) vs healthy (s): ':
        Y = line.split("\t")
        break

for i in range(0, len(Y)):
    if Y[i] == '"tumor (t) vs healthy (s): S"':
        Y[i] = "0"
    elif Y[i] == '"tumor (t) vs healthy (s): T"':
        Y[i] = "1"
    else:
        Y[i] = 'tumor (1) vs healty (0)'
Y_series_matrix='\t'.join(Y)


try:
    D=open("series_matrix.txt",'w')
except IOError:
    print("File myfile.fa does not exist!!")

Start_Reading=False
for Original_line in f:
    line=Original_line.rstrip()
    if line=='!series_matrix_table_begin':
        Start_Reading=True
    elif Start_Reading==True and line!='!series_matrix_table_end':
        D.write(Original_line)
    elif line=='!series_matrix_table_end':
        break
D.write(Y_series_matrix)

D.close()
f.close()




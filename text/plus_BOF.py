# -*- coding: utf-8 -*-


rfname = "草薙.txt"
wfname = "草薙_bof_eof.txt"

fr = open(rfname, "r")
fw = open(wfname, "w")
lines = fr.readlines()
fr.close()

for value in lines :
    fw.write("BOF "+value[:-1]+" EOF"+"\n")

fw.close()

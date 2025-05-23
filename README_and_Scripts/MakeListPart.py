#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  6 10:59:43 2024

@author: clemieux
"""


import numpy as np
import pandas as pd
import os
import csv





if __name__ == '__main__':
    cwd = os.getcwd() + "/README_and_Scripts/"
    
    #vector giving the nb of parallel sessions in each slot
    #a slot is a spot in the schedule where we have parallel sessions
    #we have 9 slots in the schedule, so this vector would have 9 components
    NbParallel = [4,4,4,4,4,4,4,4,4]
    #order of the plenary ralks: need to have for each a file Plxx.tex 
    PrintPlenary = True #set to true when ready with files
    PlSched = ["Kr","Fo","Ow","Ol","Oa","Ku","Go","La","30"]
    rows = []
    SessionNumber = 0
    SlotNumber = 0
    #counter for the nb of sessions in a slot
    WhichSess = 0
    NewSlot = True
    #keeps track of how many talks have been listed so far in the schedule
    NbTalkListed = 0
    StartListTalk = False
    
    fpart = open(f"{cwd}Participants.tex",'w')
  #  print("\\begin{sideways}\n\\begin{tabularx}{\\textheight}{l*{\\numcols}{|Y}}",file=fsched)
    
    #we assume SessionList contains the list of sessions in order of time, with sessions happening at the same
    #time ordered by "room number", e.g., columns in the schedule
    ##################IMOPORTANT
    print("\\chapter{List of Participants (as of July 5)}\n",file=fpart)
    print("\\setlength{\columnsep}{1cm}\n",file=fpart)
    print("\\begin{multicols}{2}\n",file=fpart)
    print("\\small\\raggedright\n",file=fpart)
    with open(f"{cwd}PARTICIPANTSJULY5.csv", 'r') as file:
        reader= csv.reader(file, delimiter=',')
        for val in reader:
            partstrng = "\\participant{"+val[2]+"}"
            print(partstrng,file=fpart)
            print("{",val[4],"}",file=fpart)
            print("{",val[3],"}",file=fpart)
            try:
              label1="{"+str(val[5])+"}"
              print(label1,file=fpart)
            except:
                pass
            try:
              label2="{"+str(val[6])+"}"
              print(label2,file=fpart)
            except:
                pass      
            
    print("\\end{multicols}\n",file=fpart) 
    fpart.close()        
            
        
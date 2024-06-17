import os
from src.osim_utilities import readMotionFile

MainDatapath = 'C:/Users/mat950/OneDrive - Vrije Universiteit Amsterdam/Onderwijs/MRP/Timo_Mirthe/Data/OpenCap/OpenSimData'
datafile = os.path.join(MainDatapath,'ID','hp_1.sto')
id_dat = readMotionFile(datafile)
print('test')
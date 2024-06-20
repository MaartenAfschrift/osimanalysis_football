import opensim as osim
from src.kinematic_analyses import bodykinematics
from src.osim_utilities import osim_subject
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Qt5Agg') # interactive backend for matplotlib figures

# flow control
BoolCompute = True

# path information
MainDatapath = 'C:/Users/mat950/OneDrive - Vrije Universiteit Amsterdam/Onderwijs/MRP/Timo_Mirthe/Data/OpenCap/OpenSimData'
modelpath = os.path.join(MainDatapath,'Model','LaiUhlrich2022_scaled.osim')

# init an opensim subject data processor
my_subject = osim_subject(modelpath)

# set the inverse kinematics files
ik_directory = os.path.join(MainDatapath, 'Kinematics')
my_subject.set_ikfiles_fromfolder(ik_directory)

# compute things
#    - general settings files
general_id_settings = os.path.join(MainDatapath, 'settings', 'ID_settings.xml')
forces_settings = os.path.join(MainDatapath, 'settings', 'loads_rightleg.xml')
my_subject.set_general_id_settings(general_id_settings)
my_subject.set_generic_external_loads(forces_settings)

#   - run computations
 # compute BK and read results
my_subject.read_ikfiles()
my_subject.read_bkfiles()
#my_subject.compute_bodykin(boolRead = True)
my_subject.create_extloads_soccer() # create external load files
my_subject.compute_inverse_dynamics(boolRead = True) # solve id based kinematics and loads

#   - print file with instance that person hits the ball
file_dt_hit_ball = os.path.join(MainDatapath, 'Timing_hitball.csv')
my_subject.print_file_timinghit_ball(file_dt_hit_ball)

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
#my_subject.compute_bodykin(boolRead = True) # compute BK and read results
my_subject.read_ikfiles()
my_subject.read_bkfiles()
#my_subject.create_extloads_soccer() # create external load files
#my_subject.compute_inverse_dynamics(boolRead = True) # solve id based kinematics and loads

#   - print file with instance that person hits the ball
file_dt_hit_ball = os.path.join(MainDatapath, 'Timing_hitball.csv')
my_subject.print_file_timinghit_ball(file_dt_hit_ball)









# # Flow control
# flow_runid = False
# flow_runbk = False
# flow_onefile = True # read only one file to speed up debugging intial code
#
# # general information
# MainDatapath = 'C:/Users/mat950/OneDrive - Vrije Universiteit Amsterdam/Onderwijs/MRP/Timo_Mirthe/Data/OpenCap/OpenSimData'
# modelpath = os.path.join(MainDatapath,'Model','LaiUhlrich2022_scaled.osim')
#
# # get information about all files in the folder
# ik_directory = os.path.join(MainDatapath, 'Kinematics')
# # find all .mot trials in the IK folder
# ik_files = []
# # Iterate over all files in the directory
# BoolReadFiles = True
# for file in os.listdir(ik_directory):
#     # Check if the file ends with .mot
#     if file.endswith('.mot') and BoolReadFiles:
#         ik_files.append(os.path.join(ik_directory, file))
#         if flow_onefile:
#             BoolReadFiles = False
#
# # read opensim data
# # it would be nice to have one class that stores all opensim information for a subject (like subject on a disk in addbiomechanics)
#
# # we want to estimate the force from the ball on the person using the assumption of a constant force and no energy loss during collision
#
# # We need to run bodykinematics to compute the kinetic energy of the leg (and potential energy of gravity)
# my_subject = osim_subject(modelpath, ik_files)
#
#
# # compute kinetic energy in each body segment
# # Ekin_trials = my_subject.compute_Ekin_bodies()
#
# # compute linear impulse of all bodies
# impulse_trials = my_subject.compute_linear_impulse_bodies()
#
# # create external loads files for inverse dynamics
# my_subject.create_extloads_soccer(dt_hit = 0.08, mbal = 0.450)
#
# # compute inverse dynamics
# general_id_settings = os.path.join(MainDatapath, 'settings', 'ID_settings.xml')
# forces_settings = os.path.join(MainDatapath, 'settings', 'loads_rightleg.xml')
# my_subject.set_general_id_settings(general_id_settings)
# my_subject.set_generic_external_loads(forces_settings)
# my_subject.compute_inverse_dynamics()


#
# # set timing hitting ball here
# dt = 0.06 # 0.08 to 0.1 seems to be a realistic value  (http://people.umass.edu/~gc/protected/Robertson_InverseDynamics2D.pdf)
# thit = 3.829
# mbal = 0.450
#
#
# # convert some things to a dataframe
# itrial = 0
# ik_sel = my_subject.ikdat[itrial]
# ik_header = my_subject.ik_labels
# df_ik = pd.DataFrame(ik_sel, columns = ik_header)
# # df_ekin = pd.DataFrame(Ekin_trials[itrial], columns=my_subject.bodynames)
# df_pos = pd.DataFrame(my_subject.bk_pos[itrial], columns=my_subject.bk_header)
# df_vel = pd.DataFrame(my_subject.bk_vel[itrial], columns=my_subject.bk_header)
#
# # get linear impulse of the foot
# p_footR = impulse_trials[itrial][:,my_subject.bodynames.index('calcn_r'), 0]
# p_legR = impulse_trials[itrial][:,my_subject.bodynames.index('calcn_r'), 0] +\
#          impulse_trials[itrial][:,my_subject.bodynames.index('tibia_r'), 0] + \
#          impulse_trials[itrial][:,my_subject.bodynames.index('femur_r'), 0]
#
# # get maximal velocity of the right foot (and plot this)
# it0 = np.where(df_ik.time>=thit)[0][0]
# itend = np.where(df_ik.time>=(thit+dt))[0][0]
# v_ball_post = (p_footR[it0] - p_footR[itend])/mbal
# v_ball_post_leg = (p_legR [it0] - p_legR[itend])/mbal
#
#
# plt.figure()
# plt.subplot(1,2,1)
# plt.plot(df_ik.time,df_vel.calcn_r_X)
# plt.vlines(thit,-2,20 ,color = 'black')
# plt.hlines(v_ball_post,df_ik.time.iloc[0],df_ik.time.iloc[-1], color = 'green')
# plt.hlines(v_ball_post_leg,df_ik.time.iloc[0],df_ik.time.iloc[-1], color = 'red')
# plt.subplot(1,2,2)
# plt.plot(df_ik.time,p_footR )
# plt.vlines(thit,-2,20 ,color = 'black')
#
#
# # generate force file acting from ball on foot
# nfr = len(df_ik.time)
# dat_ballFoot = np.zeros([nfr, 10])
# dat_ballFoot[:, 0] = df_ik.time
# dat_ballFoot[range(it0,itend), 1] = -(p_footR[it0] - p_footR[itend])/dt
# dat_ballFoot[:, 4] = df_pos.calcn_r_X
# dat_ballFoot[:, 5] = df_pos.calcn_r_Y
# dat_ballFoot[:, 6] = df_pos.calcn_r_Z
# dat_headers = ['time', 'ball_force_vx','ball_force_vy','ball_force_vz', 'ball_force_px','ball_force_py',
#             'ball_force_pz','ground_torque_x','ground_torque_y','ground_torque_z']
# forcesfilename = os.path.join(MainDatapath,'Loads',Path(ik_files[0]).stem + '.sto')
# generate_mot_file(dat_ballFoot, dat_headers, forcesfilename)
#
# # test solve inverse dynamics
# # general settings
# general_id_settings = os.path.join(MainDatapath, 'settings', 'ID_settings.xml')
# forces_settings = os.path.join(MainDatapath, 'settings', 'loads_rightleg.xml')
# # make folder for ID settings if needed
# id_directory = os.path.join(MainDatapath, 'ID')
# if not os.path.isdir(id_directory):
#     os.mkdir(id_directory)
# output_settings = os.path.join(MainDatapath, 'ID', 'settings')
# if not os.path.isdir(output_settings):
#     os.mkdir(output_settings)
#
# # run inverse dynamics
# idyn = InverseDynamics(model_input=modelpath, xml_input=general_id_settings, xml_output=output_settings,
#                        mot_files=ik_files[itrial], sto_output=id_directory, xml_forces = forces_settings,
#                        forces_dir = os.path.join(MainDatapath,'Loads'))
#
#
#
#
# plt.show()


# if flow_runbk:
#     bkdir = os.path.join(MainDatapath,'BK')
#     if not os.path.isdir(bkdir):
#         os.mkdir(bkdir)
#     bodykinematics(modelpath, bkdir, ik_files)


# if flow_runid:
#     # general settings
#     general_id_settings = os.path.join(MainDatapath, 'settings', 'ID_settings.xml')
#     # make folder for ID settings if needed
#     id_directory = os.path.join(MainDatapath,'ID')
#     if not os.path.isdir(id_directory):
#         os.mkdir(id_directory)
#     output_settings = os.path.join(MainDatapath,'ID','settings')
#     if not os.path.isdir(output_settings):
#         os.mkdir(output_settings)
#     # run inverse dynamics
#     idyn = InverseDynamics(model_input = modelpath, xml_input = general_id_settings, xml_output= output_settings,
#                            mot_files = ik_files, sto_output= id_directory)
#     print('inverse dynamics solved for all trials')
#
#
#
#

# # change in kinetic energy during time that perons hits te ball
# Ekin_leg_r = df_ekin.femur_r + df_ekin.tibia_r + df_ekin.calcn_r
# Ekin_leg_l = df_ekin.femur_l + df_ekin.tibia_l + df_ekin.calcn_l
# it0 = np.where(df_ik.time>=thit)[0][0]
# itend = np.where(df_ik.time>=(thit+dt))[0][0]
# Delta_Ekin = Ekin_leg_l[itend] - Ekin_leg_r[it0] # change in kinetic energy
#
# # change in impulse left and right leg
# bodies_right_leg = ['femur_r', 'tibia_r', 'calcn_r']
# impulse_all = impulse_trials[itrial]
# nfr = len(impulse_all)
# impulse_leg_r = np.zeros([nfr, 3])
# for body in bodies_right_leg:
#     icol = my_subject.bodynames.index(body)
#     impulse_leg_r = impulse_leg_r + np.squeeze(impulse_all[:,icol,:])
#
# # total impulse
# total_impulse = np.sum(impulse_all, axis = 1)
#
# Delta_impulse = total_impulse[itend,:]- total_impulse[it0,:]
# mbal = 0.450
# v_bal = -Delta_impulse/mbal
#
#
#
# # visual check is that the persons hits the ball at t = 3.838 s
#
# plt.figure()
# plt.subplot(3, 2, 1)
# plt.plot(df_ik.time,df_ekin.femur_r)
# plt.plot(df_ik.time,df_ekin.tibia_r)
# plt.plot(df_ik.time,df_ekin.calcn_r)
# plt.plot(df_ik.time,Ekin_leg_r)
# plt.vlines(thit,0,np.max(Ekin_leg_r) )
# plt.vlines(thit+dt,0,np.max(Ekin_leg_r) )
# plt.legend(['femur, tibia','foot','totalLeg-r'])
# plt.title('right leg ')
#
# plt.subplot(3, 2, 2)
# plt.plot(df_ik.time,df_ekin.femur_l)
# plt.plot(df_ik.time,df_ekin.tibia_l)
# plt.plot(df_ik.time,df_ekin.calcn_l)
# plt.plot(df_ik.time,Ekin_leg_l)
# plt.legend(['femur, tibia','foot','totalLeg-l'])
# plt.title(my_subject.ikfiles[itrial].stem)
#
# plt.subplot(3,2,3)
# plt.plot(df_ik.time,df_pos.calcn_l_X- df_pos.pelvis_X)
# plt.plot(df_ik.time,df_pos.calcn_r_X- df_pos.pelvis_X)
# plt.vlines(thit,0,np.max(df_pos.calcn_r_X- df_pos.pelvis_X) )
# plt.vlines(thit+dt,0,np.max(df_pos.calcn_r_X- df_pos.pelvis_X) )
#
# plt.subplot(3,2,4)
# plt.plot(df_ik.time,df_vel.calcn_l_X)
# plt.plot(df_ik.time,df_vel.calcn_r_X)
# plt.vlines(thit,-2,20 )
# plt.vlines(thit+dt,-2,20 )
#
# acc_foot_l = central_difference(df_ik.time,df_vel.calcn_l_X)
# acc_foot_r = central_difference(df_ik.time,df_vel.calcn_r_X)
# plt.subplot(3,2,5)
# plt.plot(df_ik.time,acc_foot_l)
# plt.plot(df_ik.time,acc_foot_r)
# plt.vlines(thit,-2,200 )
# plt.vlines(thit+dt,-2,200 )
# print('test')
#
# plt.subplot(3,2,6)
# plt.plot(df_ik.time,impulse_leg_r[:,0])
# plt.vlines(thit,-2,100, colors='black')
# plt.vlines(thit+dt,-2,100, colors='black' )
# plt.show()
# # trial 1
# itrial = 0
# pos = my_subject.bk_pos[itrial]
# vel = my_subject.bk_vel[itrial]
# bk_labels = my_subject.bk_header
# df_pos = pd.DataFrame(pos, columns=bk_labels )
# df_vel = pd.DataFrame(pos, columns=bk_labels )
#
# # kinetic energy
# nbodies = my_subject.model.getBodySet().getSize()
# bodyset = my_subject.model.getBodySet()
# nfr = len(df_pos.time)
# Ekin = np.full([nfr, nbodies], np.nan)
# for i in range(0,nbodies):
#
#     # get inertia tensor of opensim body in local coordinate system
#     I_osim_Mom = bodyset.get(i).getInertia().getMoments()
#     I_osim_Prod = bodyset.get(i).getInertia().getProducts()
#     I_body = np.zeros([3,3])
#     I_body[0, 0] = I_osim_Mom.get(0)
#     I_body[1, 1] = I_osim_Mom.get(1)
#     I_body[2, 2] = I_osim_Mom.get(0)
#     I_body[1, 0]= I_osim_Prod.get(0) # why -1 ?
#     I_body[0, 1]= I_osim_Prod.get(0)
#     I_body[0, 2]= I_osim_Prod.get(1)
#     I_body[2, 0]= I_osim_Prod.get(1)
#     I_body[2, 1]= I_osim_Prod.get(2)
#     I_body[1, 2]= I_osim_Prod.get(2)
#     m = bodyset.get(i).getMass()
#
#     # compute angular momentum at each frame
#     bodyName = bodyset.get(i).getName()
#     nfr = len(df_pos.time)
#     fi_dot = np.zeros([nfr, 3])
#     fi_dot[:, 0] = df_vel[bodyName + '_Ox']
#     fi_dot[:, 1] = df_vel[bodyName + '_Oy']
#     fi_dot[:, 2] = df_vel[bodyName + '_Oz']
#
#     fi = np.zeros([nfr, 3])
#     fi[:, 0] = df_pos[bodyName + '_Ox']
#     fi[:, 1] = df_pos[bodyName + '_Oy']
#     fi[:, 2] = df_pos[bodyName + '_Oz']
#
#     r_dot = np.zeros([nfr, 3])
#     r_dot[:, 0] = df_vel[bodyName + '_X']
#     r_dot[:, 1] = df_vel[bodyName + '_Y']
#     r_dot[:, 2] = df_vel[bodyName + '_Z']
#
#     # inertia in world coordinate system
#     for t in range(0, nfr):
#         T_Body = transform(fi[t,0], fi[t,1], fi[t,2])
#         I_world = T_Body.T * I_body * T_Body
#
#         # rotational kinetic energy
#         Ek_rot = 0.5*np.dot(np.dot(fi_dot[t,:].T,I_world),fi_dot[t,:])
#
#         # translational kinetic energy
#         Ek_transl = 0.5 * m * np.dot(r_dot[t,:].T, r_dot[t,:])
#
#         # total kinetic energy
#         Ekin[t, i] = Ek_rot + Ek_transl


# plt.figure()
# plt.plot(Ekin)
# plt.show()








# test some simple computations for collision with ball

# compute kinetic energy of left and right leg










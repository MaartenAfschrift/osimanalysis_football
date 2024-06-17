import os
from pathlib import Path
import opensim as osim
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from src.inverse_dynamics import InverseDynamics
from src.kinematic_analyses import bodykinematics

def readMotionFile(filename):
    """ Reads OpenSim .sto files.
    Parameters
    ----------
    filename: absolute path to the .sto file
    Returns
    -------
    header: the header of the .sto
    labels: the labels of the columns
    data: an array of the data
    """

    if not os.path.exists(filename):
        print('file do not exists')

    file_id = open(filename, 'r')

    # read header
    next_line = file_id.readline()
    header = [next_line]
    nc = 0
    nr = 0
    while not 'endheader' in next_line:
        if 'datacolumns' in next_line:
            nc = int(next_line[next_line.index(' ') + 1:len(next_line)])
        elif 'datarows' in next_line:
            nr = int(next_line[next_line.index(' ') + 1:len(next_line)])
        elif 'nColumns' in next_line:
            nc = int(next_line[next_line.index('=') + 1:len(next_line)])
        elif 'nRows' in next_line:
            nr = int(next_line[next_line.index('=') + 1:len(next_line)])

        next_line = file_id.readline()
        header.append(next_line)

    # process column labels
    next_line = file_id.readline()
    if next_line.isspace() == True:
        next_line = file_id.readline()

    labels = next_line.split()

    # get data
    data = pd.DataFrame(columns=labels)
    data = np.ndarray([nr, len(labels)])
    for i in range(1, nr + 1):
        d = [float(x) for x in file_id.readline().split()]
        data[i-1,:] = d
    file_id.close()
    dat = pd.DataFrame(data, columns = labels)

    return dat

# general class to do bathc processing in opensim
# This class contains all information for a subject.
# This one currently assumtes that you provide the path
# to inverse kinematics files as input. I'll change this
# in the future (use marker coordinates as input)

class osim_subject:
    def __init__(self, modelfile, maindir = []):

        # some default settings
        if len(maindir) == 0:
            # assumes that main datadir one folder up the tree from the modelfile path
            self.maindir = Path(modelfile).parents[1]
        else:
            self.maindir = []

        # variables class
        self.modelpath = modelfile # path to (scaled) opensim model
        self.ikfiles = [] # path to ikfiles
        self.nfiles = [] # number of files
        self.filenames = [] # names of the files
        self.general_id_settings = [] # generic id settings
        self.extload_settings = [] # generic external load settings
        self.ext_loads_dir = self.maindir.joinpath('Loads')
        self.id_directory = self.maindir.joinpath('ID')
        self.bodykin_folder = self.maindir.joinpath('BK')
        self.extload_files = []
        self.id_dat = [] # inverse dynamics data
        self.bk_pos = [] # bodykinematics data -- position
        self.bk_vel = [] # bodykinematics data -- velocity

        # open model
        self.model = osim.Model(self.modelpath)

        # read some default things from the model
        [self.modelmass, self.bodymass, self.bodynames]= self.getmodelmass()

    def set_ikfiles_fromfolder(self, ikfolder):

        # find all .mot trials in the IK folder
        ik_files = []
        for file in os.listdir(ikfolder):
            # Check if the file ends with .mot
            if file.endswith('.mot'):
                ik_files.append(os.path.join(ikfolder, file))
        # convert to Path objects
        if not isinstance(self.ikfiles[0], Path):
            self.ikfiles = [Path(i) for i in self.ikfiles]
        # get number of files
        self.nfiles = len(self.ikfiles)

        # get filenames based on ik files
        self.filenames = []
        for itrial in self.ikfiles:
            self.filenames.append(itrial.stem)

    def set_ikfiles(self, ikfiles):
        # set the names of the IK files
        self.ikfiles = ikfiles

        # convert to a list if needed
        if not isinstance(ikfiles, list):
            self.ikfiles = [ikfiles]
        else:
            self.ikfiles = ikfiles

        # convert to Path objects
        if not isinstance(self.ikfiles[0], Path):
            self.ikfiles = [Path(i) for i in self.ikfiles]

        # get number of files
        self.nfiles = len(self.ikfiles)

        # set some default directories based on ikfiles
        self.id_directory = self.ikfiles[0].parents[1].joinpath('ID')

        # get filenames based on ik files
        self.filenames = []
        for itrial in self.ikfiles:
            self.filenames.append(itrial.stem)

    def compute_bodykin(self, boolRead = True):
        # function to compute bodykinematics
        bkdir = self.bodykin_folder
        if not os.path.isdir(bkdir):
            os.mkdir(bkdir)
        bodykinematics(self.modelpath, bkdir, self.ikfiles)
        # read the bk files
        if boolRead:
            self.read_bkfiles()

    # ----- inverse kinematics -------
    def read_ikfiles(self):
        # read ik files
        self.ikdat = []
        self.ik_labels = []
        for itrial in self.ikfiles:
            ik_data = self.read_ik(itrial)
            self.ikdat.append(ik_data)

    def read_ik(self, ik_file):
        if ik_file.exists():
            id_data = readMotionFile(ik_file)
        else:
            ik_data = []
            print('could find read file ', ik_file)
        return(ik_data)

    # ----- bodykinematics -------
    def read_bkfiles(self):
        # read bodykinematics files
        self.bk_pos = []
        self.bk_vel = []
        for itrial in self.ikfiles:
            trialname = f'{itrial.stem}'
            [bk_pos, bk_vel, bk_labels] = self.read_bodykin(self.bodykin_folder, trialname)
            self.bk_pos.append(bk_pos)
            self.bk_vel.append(bk_vel)
            self.bk_header = bk_labels

    def read_bodykin(self, bkfolder, trial_stem):
        # path to position and velocity file
        bk_pos_file = Path(os.path.join(bkfolder, trial_stem + '_BodyKinematics_pos_global.sto'))
        bk_vel_file = Path(os.path.join(bkfolder, trial_stem + '_BodyKinematics_vel_global.sto'))
        if bk_pos_file.exists():
            bk_pos = readMotionFile(bk_pos_file)
        else:
            bk_pos = []
        if bk_vel_file.exists():
            bk_vel = readMotionFile(bk_vel_file)
        else:
            bk_vel = []
            print('could find read file ', bk_vel_file)
        return(bk_pos, bk_vel)

    def getmodelmass(self):
        # read the opensim model
        nbodies = self.model.getBodySet().getSize()
        m_bodies = np.full([nbodies], np.nan)
        bodynames = []
        bodyset = self.model.get_BodySet()
        for i in range(0, nbodies):
            bodynames.append(bodyset.get(i).getName())
            m_bodies[i] = bodyset.get(i).getMass()
        m_tot = np.nansum(m_bodies)
        return(m_tot, m_bodies, bodynames)

    def compute_Ekin_bodies(self):
        print('start computation kinetic energy')
        # computes kinetic energy of all rigid bodies
        bk_header = self.bk_header
        nbodies = self.model.getBodySet().getSize()
        bodyset = self.model.getBodySet()
        Ekin_trials =[]
        for ifile in range(0, self.nfiles):
            bk_pos = self.bk_pos[ifile]
            bk_vel = self.bk_vel[ifile]
            if len(bk_pos)>0 and len(bk_vel)>0:
                # I like to work with pandas tables
                df_pos = pd.DataFrame(bk_pos, columns=bk_header)
                df_vel = pd.DataFrame(bk_vel, columns=bk_header)
                nfr = len(df_pos.time)
                Ekin = np.full([nfr, nbodies], np.nan)
                for i in range(0, nbodies):

                    # get inertia tensor of opensim body in local coordinate system
                    I_body = osim_body_I(bodyset.get(i).getInertia())
                    m = bodyset.get(i).getMass()

                    # compute angular momentum at each frame
                    bodyName = bodyset.get(i).getName()
                    nfr = len(df_pos.time)
                    fi_dot = np.zeros([nfr, 3])
                    fi_dot[:, 0] = df_vel[bodyName + '_Ox']
                    fi_dot[:, 1] = df_vel[bodyName + '_Oy']
                    fi_dot[:, 2] = df_vel[bodyName + '_Oz']

                    fi = np.zeros([nfr, 3])
                    fi[:, 0] = df_pos[bodyName + '_Ox']
                    fi[:, 1] = df_pos[bodyName + '_Oy']
                    fi[:, 2] = df_pos[bodyName + '_Oz']

                    r_dot = np.zeros([nfr, 3])
                    r_dot[:, 0] = df_vel[bodyName + '_X']
                    r_dot[:, 1] = df_vel[bodyName + '_Y']
                    r_dot[:, 2] = df_vel[bodyName + '_Z']

                    # inertia in world coordinate system
                    for t in range(0, nfr):
                        T_Body = transform(fi[t, 0], fi[t, 1], fi[t, 2])
                        I_world = T_Body.T * I_body * T_Body

                        # rotational kinetic energy
                        Ek_rot = 0.5 * np.dot(np.dot(fi_dot[t, :].T, I_world), fi_dot[t, :])

                        # translational kinetic energy
                        Ek_transl = 0.5 * m * np.dot(r_dot[t, :].T, r_dot[t, :])

                        # total kinetic energy
                        Ekin[t, i] = Ek_rot + Ek_transl
            else:
                Ekin = []
            Ekin_trials.append(Ekin)
            print('... file ' + str(ifile+1) + '/' + str(self.nfiles))
        return(Ekin_trials)

    def compute_linear_impulse_bodies(self):
        # computes the linear impulse of all bodies
        print('started with computation linear impulse of all bodies')
        nbodies = self.model.getBodySet().getSize()
        bodyset = self.model.getBodySet()
        impulse_trials =[]
        for ifile in range(0, self.nfiles):
            bk_pos = self.bk_pos[ifile]
            bk_vel = self.bk_vel[ifile]
            if len(bk_pos) > 0 and len(bk_vel) > 0:
                # I like to work with pandas tables
                nfr = len(bk_vel.time)
                linear_impulse = np.full([nfr, nbodies, 3], np.nan)
                for i in range(0, nbodies):
                    m = bodyset.get(i).getMass()
                    bodyName = bodyset.get(i).getName()
                    r_dot = np.zeros([nfr, 3])
                    r_dot[:, 0] = bk_vel[bodyName + '_X']
                    r_dot[:, 1] = bk_vel[bodyName + '_Y']
                    r_dot[:, 2] = bk_vel[bodyName + '_Z']
                    linear_impulse[:,i,:] = m * r_dot
            else:
                linear_impulse = []
            impulse_trials.append(linear_impulse)
            print('... file ' + str(ifile + 1) + '/' + str(self.nfiles))
        return(impulse_trials)

    def create_extloads_soccer(self, dt_hit = 0.1, mbal = 0.450):
        # this approach is based on the assumption that there is a conservation of linear
        # impulse of the left and right foot
        impulse_trials = self.compute_linear_impulse_bodies()
        self.extload_files = []

        for itrial in range(0, self.nfiles):
            # convert to pandas structure
            df_ik  = self.ikdat[itrial]
            df_pos = self.bk_pos[itrial]
            df_vel = self.bk_vel[itrial]

            # assumption that persons hits the ball at max velocity of the foot
            thit = df_vel.time.iloc[np.argmax(df_vel.calcn_r_X)]

            # get linear impulse of the foot
            p_footR = impulse_trials[itrial][:, self.bodynames.index('calcn_r'), 0]

            # compute release velocity ball based on assumption conservation of linear impulse
            it0 = np.where(df_ik.time >= thit)[0][0]
            itend = np.where(df_ik.time >= (thit + dt_hit))[0][0]
            v_ball_post = (p_footR[it0] - p_footR[itend]) / mbal

            # create externa loads file for soccer kick -- right leg
            nfr = len(df_ik.time)
            dat_ballFoot = np.zeros([nfr, 10])
            dat_ballFoot[:, 0] = df_ik.time
            dat_ballFoot[range(it0, itend), 1] = -(p_footR[it0] - p_footR[itend]) / dt_hit
            dat_ballFoot[:, 4] = df_pos.calcn_r_X
            dat_ballFoot[:, 5] = df_pos.calcn_r_Y
            dat_ballFoot[:, 6] = df_pos.calcn_r_Z
            dat_headers = ['time', 'ball_force_vx', 'ball_force_vy', 'ball_force_vz', 'ball_force_px', 'ball_force_py',
                           'ball_force_pz', 'ground_torque_x', 'ground_torque_y', 'ground_torque_z']
            forcesfilename = os.path.join(self.ext_loads_dir, Path(self.ikfiles[itrial]).stem + '.sto')
            generate_mot_file(dat_ballFoot, dat_headers, forcesfilename)
            self.extload_files.append(forcesfilename)

    def compute_inverse_dynamics(self, boolRead= True):
        # computes inverse dynamics for all ik files
        print('work in progress')
        # id output settings
        output_settings = os.path.join(self.id_directory, 'settings')
        if not os.path.isdir(output_settings):
            os.mkdir(output_settings)
        for itrial in range(0,self.nfiles):
            # solve inverse dynamics for this trial
            idyn = InverseDynamics(model_input=self.modelpath,
                                   xml_input=self.general_id_settings,
                                   xml_output=output_settings,
                                   mot_files=self.ikfiles[itrial],
                                   sto_output=self.id_directory,
                                   xml_forces=self.extload_settings,
                                   forces_dir=self.ext_loads_dir)
        if boolRead:
            self.read_inverse_dynamics()

    def read_inverse_dynamics(self):
        self.id_dat = []
        for itrial in range(0, self.nfiles):
                # path to position and velocity file
                id_file = Path(os.path.join(self.id_directory, self.ikfiles[itrial].stem + '.sto'))
                if id_file.exists():
                    id_dat = readMotionFile(id_file)
                else:
                    id_dat = []
                    print('could find read file ', id_file)
                self.id_dat.append(id_dat)

    def set_general_id_settings(self, general_id_settings):
        self.general_id_settings = general_id_settings

    def set_generic_external_loads(self, general_loads_settings):
        self.extload_settings = general_loads_settings

    def set_id_directory(self, id_directory):
        self.id_directory = id_directory

    def set_ext_loads_dir(self, ext_loads_dir):
        self.ext_loads_dir = self.ext_loads_dir


def transform(Rx, Ry, Rz, bool_deg_input = True):
    """
    Compute transformation matrix for x-y'-z'' intrinsic Euler rotations
    (the OpenSim convention).

    Parameters:
    Rx (float): Rotation around the x-axis in degrees.
    Ry (float): Rotation around the y-axis in degrees.
    Rz (float): Rotation around the z-axis in degrees.

    Returns:
    numpy.ndarray: The 3x3 transformation matrix.
    """
    # Convert degrees to radians
    if bool_deg_input:
        Rx = np.radians(Rx)
        Ry = np.radians(Ry)
        Rz = np.radians(Rz)

    # Compute transformation matrix elements
    R11 = np.cos(Ry) * np.cos(Rz)
    R12 = -np.cos(Ry) * np.sin(Rz)
    R13 = np.sin(Ry)
    R21 = np.cos(Rx) * np.sin(Rz) + np.sin(Rx) * np.sin(Ry) * np.cos(Rz)
    R22 = np.cos(Rx) * np.cos(Rz) - np.sin(Rx) * np.sin(Ry) * np.sin(Rz)
    R23 = -np.sin(Rx) * np.cos(Ry)
    R31 = np.sin(Rx) * np.sin(Rz) - np.cos(Rx) * np.sin(Ry) * np.cos(Rz)
    R32 = np.sin(Rx) * np.cos(Rz) + np.cos(Rx) * np.sin(Ry) * np.sin(Rz)
    R33 = np.cos(Rx) * np.cos(Ry)

    # Create the transformation matrix
    R = np.array([[R11, R12, R13],
                  [R21, R22, R23],
                  [R31, R32, R33]])

    return R

def osim_body_I(Inertia):
    # returns Inertia tensor as 3x3 nummpy ndraay based on an opensim Inertia object
    I_osim_Mom = Inertia.getMoments()
    I_osim_Prod = Inertia.getProducts()
    I_body = np.zeros([3,3])
    I_body[0, 0] = I_osim_Mom.get(0)
    I_body[1, 1] = I_osim_Mom.get(1)
    I_body[2, 2] = I_osim_Mom.get(0)
    I_body[1, 0]= I_osim_Prod.get(0)
    I_body[0, 1]= I_osim_Prod.get(0)
    I_body[0, 2]= I_osim_Prod.get(1)
    I_body[2, 0]= I_osim_Prod.get(1)
    I_body[2, 1]= I_osim_Prod.get(2)
    I_body[1, 2]= I_osim_Prod.get(2)
    return(I_body)

def generate_mot_file(data_matrix, colnames, filename):
    datarows, datacols = data_matrix.shape
    time = data_matrix[:, 0]
    range_values = [time[0], time[-1]]

    if len(colnames) != datacols:
        raise ValueError(f'Number of column names ({len(colnames)}) does not match the number of columns in the data ({datacols})')

    # Open the file for writing
    try:
        with open(filename, 'w') as fid:
            # Write MOT file header
            fid.write(f'{filename}\nnRows={datarows}\nnColumns={datacols}\n\n')
            fid.write(f'name {filename}\ndatacolumns {datacols}\ndatarows {datarows}\nrange {range_values[0]} {range_values[1]}\nendheader\n')

            # Write column names
            cols = '\t'.join(colnames) + '\n'
            fid.write(cols)

            # Write data
            for i in range(datarows):
                row = '\t'.join([f'{value:20.10f}' for value in data_matrix[i, :]]) + '\n'
                fid.write(row)

    except IOError:
        print(f'\nERROR: {filename} could not be opened for writing...\n')



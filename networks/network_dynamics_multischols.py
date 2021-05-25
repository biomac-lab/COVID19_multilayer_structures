from jax.dtypes import dtype
import jax.numpy as np
import numpy as np2
from jax import random


### Intervention functions

def morning_set_intervention_ms(Graphs_matrix, intervention_eff, hh_occupation=0.9):

    # load networks
    matrix_household = Graphs_matrix[0]
    # matrix_household[2] = [val.values[0] for val in matrix_household[2]]
    hh_row = np.asarray(np2.asarray(matrix_household[0]))
    hh_col = np.asarray(np2.asarray(matrix_household[1]))
    hh_data = np.asarray(np2.asarray(matrix_household[2]))

    lst_matrix_preschool = Graphs_matrix[1]
    lst_prschol_row = []
    lst_prschol_col = []
    lst_prschol_data = []
    for n_school, matrix_preschool in enumerate(lst_matrix_preschool):
        # matrix_preschool[2] = [val.values[0] for val in matrix_preschool[2]]
        preschl_row = np.asarray(np2.asarray(matrix_preschool[0]))
        preschl_col = np.asarray(np2.asarray(matrix_preschool[1]))
        preschl_data = np.asarray(np2.asarray(matrix_preschool[2]))
        lst_prschol_row.append(preschl_row)
        lst_prschol_col.append(preschl_col)
        lst_prschol_data.append(preschl_data)

    lst_matrix_primary = Graphs_matrix[2]
    lst_primary_row = []
    lst_primary_col = []
    lst_primary_data = []
    for n_school, matrix_primary in enumerate(lst_matrix_primary):
        # matrix_primary[2] = [val.values[0] for val in matrix_primary[2]]
        preschl_row = np.asarray(np2.asarray(matrix_primary[0]))
        preschl_col = np.asarray(np2.asarray(matrix_primary[1]))
        preschl_data = np.asarray(np2.asarray(matrix_primary[2]))
        lst_primary_row.append(preschl_row)
        lst_primary_col.append(preschl_col)
        lst_primary_data.append(preschl_data)

    lst_matrix_highschool = Graphs_matrix[2]
    lst_highschl_row = []
    lst_highschl_col = []
    lst_highschl_data = []
    for n_school, matrix_highschl in enumerate(lst_matrix_highschool):
        # matrix_highschl[2] = [val.values[0] for val in matrix_highschl[2]]
        preschl_row = np.asarray(np2.asarray(matrix_highschl[0]))
        preschl_col = np.asarray(np2.asarray(matrix_highschl[1]))
        preschl_data = np.asarray(np2.asarray(matrix_highschl[2]))
        lst_highschl_row.append(preschl_row)
        lst_highschl_col.append(preschl_col)
        lst_highschl_data.append(preschl_data)

    matrix_work = Graphs_matrix[4]
    # matrix_work[2] = [val.values[0] for val in matrix_work[2]]
    work_row = np.asarray(np2.asarray(matrix_work[0]))
    work_col = np.asarray(np2.asarray(matrix_work[1]))
    work_data = np.asarray(np2.asarray(matrix_work[2]))

    matrix_community = Graphs_matrix[5]
    # matrix_community[2] = [val.values[0] for val in matrix_community[2]]
    comm_row = np.asarray(np2.asarray(matrix_community[0]))
    comm_col = np.asarray(np2.asarray(matrix_community[1]))
    comm_data = np.asarray(np2.asarray(matrix_community[2]))

    # turn off school and work layers
    preschl_data_set = 0*preschl_data
    primary_data_set = 0*primary_data
    highschl_data_set = 0*highschl_data
    work_data_set = 0*work_data

    # turn on portions of households and community
    hh_occupation_intervention = hh_occupation*(1-intervention_eff)
    comm_occupation = 1-hh_occupation
    comm_occupation_intervention = comm_occupation*(1-intervention_eff)

    length = int(hh_data.shape[0]/2)
    hh_data_select = np.repeat( random.bernoulli(random.PRNGKey(0),p=(hh_occupation_intervention),
                                              shape=(length,)), 2) 
    hh_data_set = hh_data_select.reshape(hh_data_select.shape[0],1)*hh_data

    length = int(comm_data.shape[0]/2)
    comm_data_select = np.repeat( random.bernoulli(random.PRNGKey(0),p=(comm_occupation_intervention),
                                              shape=(length,)), 2) 
    comm_data_set = comm_data_select.reshape(comm_data_select.shape[0],1)*comm_data

    # create conections
    args_ps = (hh_data_set,preschl_data_set,primary_data_set,highschl_data_set,work_data_set,comm_data_set)
    ps = np.concatenate(args_ps); ps = ps.reshape(ps.shape[0],)
    args_rows = (hh_row,preschl_row,primary_row,highschl_row,work_row,comm_row)
    rows = np.concatenate(args_rows)
    args_cols = (hh_col,preschl_col,primary_col,highschl_col,work_col,comm_col)

    cols = np.concatenate(args_cols)

    w = [rows.astype(np.int32),cols.astype(np.int32),ps]

    return w


def day_set_intervention(Graphs_matrix, intervention_eff, schl_occupation, work_occupation,
                         schl_altern=False, hh_occupation=0.3, comm_occupation=0.2):
    # load networks
    matrix_household = Graphs_matrix[0]
    # matrix_household[2] = [val.values[0] for val in matrix_household[2]]
    hh_row = np.asarray(np2.asarray(matrix_household[0]))
    hh_col = np.asarray(np2.asarray(matrix_household[1]))
    hh_data = np.asarray(np2.asarray(matrix_household[2]))

    matrix_preschool = Graphs_matrix[1]
    # matrix_preschool[2] = [val.values[0] for val in matrix_preschool[2]]
    preschl_row = np.asarray(np2.asarray(matrix_preschool[0]))
    preschl_col = np.asarray(np2.asarray(matrix_preschool[1]))
    preschl_data = np.asarray(np2.asarray(matrix_preschool[2]))

    matrix_primary = Graphs_matrix[2]
    # matrix_primary[2] = [val.values[0] for val in matrix_primary[2]]
    primary_row = np.asarray(np2.asarray(matrix_primary[0]))
    primary_col = np.asarray(np2.asarray(matrix_primary[1]))
    primary_data = np.asarray(np2.asarray(matrix_primary[2]))

    matrix_highschool = Graphs_matrix[3]
    # matrix_highschool[2] = [val.values[0] for val in matrix_highschool[2]]
    highschl_row = np.asarray(np2.asarray(matrix_highschool[0]))
    highschl_col = np.asarray(np2.asarray(matrix_highschool[1]))
    highschl_data = np.asarray(np2.asarray(matrix_highschool[2]))

    matrix_work = Graphs_matrix[4]
    # matrix_work[2] = [val.values[0] for val in matrix_work[2]]
    work_row = np.asarray(np2.asarray(matrix_work[0]))
    work_col = np.asarray(np2.asarray(matrix_work[1]))
    work_data = np.asarray(np2.asarray(matrix_work[2]))

    matrix_community = Graphs_matrix[5]
    # matrix_community[2] = [val.values[0] for val in matrix_community[2]]
    comm_row = np.asarray(np2.asarray(matrix_community[0]))
    comm_col = np.asarray(np2.asarray(matrix_community[1]))
    comm_data = np.asarray(np2.asarray(matrix_community[2]))

    # turn off portions of households and community
    hh_occupation_intervention = hh_occupation*(1-intervention_eff)
    comm_occupation = 1-hh_occupation
    comm_occupation_intervention = comm_occupation*(1-intervention_eff)

    length = int(hh_data.shape[0]/2)
    hh_data_select = np.repeat( random.bernoulli(random.PRNGKey(0),p=(hh_occupation_intervention),
                                              shape=(length,)), 2) 
    hh_data_set = hh_data_select.reshape(hh_data_select.shape[0],1)*hh_data

    length = int(comm_data.shape[0]/2)
    comm_data_select = np.repeat( random.bernoulli(random.PRNGKey(0),p=(comm_occupation_intervention),
                                              shape=(length,)), 2) 
    comm_data_set = comm_data_select.reshape(comm_data_select.shape[0],1)*comm_data

    # turn off portions of school and work layers
    if schl_occupation == 0:    
        preschl_data_set = 0*preschl_data
        primary_data_set = 0*primary_data
        highschl_data_set = 0*highschl_data
    elif schl_occupation == 1.0:
        preschl_data_set = preschl_data
        primary_data_set = primary_data
        highschl_data_set = highschl_data
    else:
        length = int(preschl_data.shape[0]/2)
        preschl_data_select = np.repeat( random.bernoulli(random.PRNGKey(0),p=(schl_occupation),
                                                shape=(length,)), 2) 
        preschl_data_set = preschl_data_select.reshape(preschl_data_select.shape[0],1)*preschl_data

        length = int(primary_data.shape[0]/2)
        primary_data_select = np.repeat( random.bernoulli(random.PRNGKey(0),p=(schl_occupation),
                                                shape=(length,)), 2) 
        primary_data_set = primary_data_select.reshape(primary_data_select.shape[0],1)*primary_data

        length = int(highschl_data.shape[0]/2)
        highschl_data_select = np.repeat( random.bernoulli(random.PRNGKey(0),p=(schl_occupation),
                                                shape=(length,)), 2) 
        highschl_data_set = highschl_data_select.reshape(highschl_data_select.shape[0],1)*highschl_data


        
    # work_occuption_intervention = 1-intervention_eff
    length = int(work_data.shape[0]/2)
    work_data_select = np.repeat( random.bernoulli(random.PRNGKey(0),p=(work_occupation),
                                            shape=(length,)), 2) 
    work_data_set = work_data_select.reshape(work_data_select.shape[0],1)*work_data
    if work_occupation == 0:
        work_data_set = 0*work_data     # if work offices are fully closed

    # create conections
    args_ps = (hh_data_set,preschl_data_set,primary_data_set,highschl_data_set,work_data_set,comm_data_set)
    ps = np.concatenate(args_ps); ps = ps.reshape(ps.shape[0],)
    args_rows = (hh_row,preschl_row,primary_row,highschl_row,work_row,comm_row)
    rows = np.concatenate(args_rows)
    args_cols = (hh_col,preschl_col,primary_col,highschl_col,work_col,comm_col)

    cols = np.concatenate(args_cols)

    w = [rows.astype(np.int32),cols.astype(np.int32),ps]

    return w


def night_set_intervention(Graphs_matrix, intervention_eff, hh_occupation=0.7):

    # load networks
    matrix_household = Graphs_matrix[0]
    # matrix_household[2] = [val.values[0] for val in matrix_household[2]]
    hh_row = np.asarray(np2.asarray(matrix_household[0]))
    hh_col = np.asarray(np2.asarray(matrix_household[1]))
    hh_data = np.asarray(np2.asarray(matrix_household[2]))

    matrix_preschool = Graphs_matrix[1]
    # matrix_preschool[2] = [val.values[0] for val in matrix_preschool[2]]
    preschl_row = np.asarray(np2.asarray(matrix_preschool[0]))
    preschl_col = np.asarray(np2.asarray(matrix_preschool[1]))
    preschl_data = np.asarray(np2.asarray(matrix_preschool[2]))

    matrix_primary = Graphs_matrix[2]
    # matrix_primary[2] = [val.values[0] for val in matrix_primary[2]]
    primary_row = np.asarray(np2.asarray(matrix_primary[0]))
    primary_col = np.asarray(np2.asarray(matrix_primary[1]))
    primary_data = np.asarray(np2.asarray(matrix_primary[2]))

    matrix_highschool = Graphs_matrix[3]
    # matrix_highschool[2] = [val.values[0] for val in matrix_highschool[2]]
    highschl_row = np.asarray(np2.asarray(matrix_highschool[0]))
    highschl_col = np.asarray(np2.asarray(matrix_highschool[1]))
    highschl_data = np.asarray(np2.asarray(matrix_highschool[2]))

    matrix_work = Graphs_matrix[4]
    # matrix_work[2] = [val.values[0] for val in matrix_work[2]]
    work_row = np.asarray(np2.asarray(matrix_work[0]))
    work_col = np.asarray(np2.asarray(matrix_work[1]))
    work_data = np.asarray(np2.asarray(matrix_work[2]))

    matrix_community = Graphs_matrix[5]
    # matrix_community[2] = [val.values[0] for val in matrix_community[2]]
    comm_row = np.asarray(np2.asarray(matrix_community[0]))
    comm_col = np.asarray(np2.asarray(matrix_community[1]))
    comm_data = np.asarray(np2.asarray(matrix_community[2]))

    # turn off school and work layers
    preschl_data_set = 0*preschl_data
    primary_data_set = 0*primary_data
    highschl_data_set = 0*highschl_data
    work_data_set = 0*work_data

    # turn on portions of households and community
    hh_occupation_intervention = hh_occupation*(1-intervention_eff)
    comm_occupation = 1-hh_occupation
    comm_occupation_intervention = comm_occupation*(1-intervention_eff)

    length = int(hh_data.shape[0]/2)
    hh_data_select = np.repeat( random.bernoulli(random.PRNGKey(0),p=(hh_occupation_intervention),
                                              shape=(length,)), 2) 
    hh_data_set = hh_data_select.reshape(hh_data_select.shape[0],1)*hh_data

    length = int(comm_data.shape[0]/2)
    comm_data_select = np.repeat( random.bernoulli(random.PRNGKey(0),p=(comm_occupation_intervention),
                                              shape=(length,)), 2) 
    comm_data_set = comm_data_select.reshape(comm_data_select.shape[0],1)*comm_data

    # create conections
    args_ps = (hh_data_set,preschl_data_set,primary_data_set,highschl_data_set,work_data_set,comm_data_set)
    ps = np.concatenate(args_ps); ps = ps.reshape(ps.shape[0],)
    args_rows = (hh_row,preschl_row,primary_row,highschl_row,work_row,comm_row)
    rows = np.concatenate(args_rows)
    args_cols = (hh_col,preschl_col,primary_col,highschl_col,work_col,comm_col)
    cols = np.concatenate(args_cols)

    w = [rows.astype(np.int32),cols.astype(np.int32),ps]

    return w


def create_day_intervention_dynamics(Graphs_matrix,Tmax,total_steps,schools_day_open,interv_glob,
                                     schl_occupation,work_occupation,partitions=[8,8,8]):
    '''
    A day is devided in 3 partitions with consists of sets of hours over a day
    partition[0] -> morning: only a % of households and community is activated
    partition[1] -> evening: only work and school layers are activated
    partition[2] -> night: only a % of households and community is activated
    delta_t      -> steps over a day
    '''
    # Hours distribution in a day
    if sum(partitions) != 24:
        print('Partitions must sum the total of hours in a day (24h)')

    steps_per_days = int(total_steps/Tmax)
    
    m_day = int(steps_per_days*(partitions[0]/24))
    e_day = int(steps_per_days*(partitions[1]/24))
    n_day = int(steps_per_days*(partitions[2]/24))
    days_intervals = [m_day, e_day, n_day]

    m_w_interv = morning_set_intervention_ms(Graphs_matrix,interv_glob)
    e_w_interv_schl_close = day_set_intervention(Graphs_matrix,interv_glob,schl_occupation=0,work_occupation=work_occupation)
    e_w_interv_schl_open  = day_set_intervention(Graphs_matrix,interv_glob,schl_occupation=schl_occupation,work_occupation=work_occupation)
    n_w_interv = night_set_intervention(Graphs_matrix,interv_glob)

    w_interv_intervals_schl_close = [m_w_interv,e_w_interv_schl_close,n_w_interv]
    w_interv_intervals_schl_open  = [m_w_interv,e_w_interv_schl_open,n_w_interv]

    sim_intervals = []  # iterations per network set w
    sim_ws        = []  # networks per iteration
    for d in range(Tmax):
        if d < schools_day_open:
            sim_intervals.extend(days_intervals)
            sim_ws.extend(w_interv_intervals_schl_close)
        else:
            sim_intervals.extend(days_intervals)
            sim_ws.extend(w_interv_intervals_schl_open)

    return sim_intervals, sim_ws


def create_day_intervention_altern_schools_dynamics(Graphs_matrix,Tmax,total_steps,schools_day_open,
                                interv_glob,schl_occupation,work_occupation,partitions=[8,8,8]):

    '''
    A day is devided in 3 partitions with consists of sets of hours over a day
    partition[0] -> morning: only a % of households and community is activated
    partition[1] -> evening: only work and school layers are activated
    partition[2] -> night: only a % of households and community is activated
    delta_t      -> steps over a day
    '''
    # Hours distribution in a day
    if sum(partitions) != 24:
        print('Partitions must sum the total of hours in a day (24h)')

    steps_per_days = int(total_steps/Tmax)
    
    m_day = int(steps_per_days*(partitions[0]/24))
    e_day = int(steps_per_days*(partitions[1]/24))
    n_day = int(steps_per_days*(partitions[2]/24))
    days_intervals = [m_day, e_day, n_day]

    m_w_interv = morning_set_intervention_ms(Graphs_matrix,interv_glob)
    e_w_interv_schl_close = day_set_intervention(Graphs_matrix,interv_glob,schl_occupation=0,work_occupation=work_occupation)
    e_w_interv_schl_open_set1  = day_set_intervention(Graphs_matrix,interv_glob,schl_occupation,work_occupation)
    e_w_interv_schl_open_set2  = day_set_intervention(Graphs_matrix,interv_glob,schl_occupation,work_occupation)
    n_w_interv = night_set_intervention(Graphs_matrix,interv_glob)

    w_interv_intervals_schl_close = [m_w_interv,e_w_interv_schl_close,n_w_interv]
    w_interv_intervals_schl_open_set1  = [m_w_interv,e_w_interv_schl_open_set1,n_w_interv]
    w_interv_intervals_schl_open_set1  = [m_w_interv,e_w_interv_schl_open_set2,n_w_interv]   

    altern_period = 8   # days
    days_1 = 0 
    days_2 = 0

    sim_intervals = []  # iterations per network set w
    sim_ws        = []  # networks per iteration
    for i, d in enumerate(range(Tmax)):
        if d < schools_day_open:
            sim_intervals.extend(days_intervals)
            sim_ws.extend(w_interv_intervals_schl_close)
        else:
            if days_1 < altern_period and days_2 == 0:
                sim_intervals.extend(days_intervals)
                sim_ws.extend(w_interv_intervals_schl_open_set1)
                days_1 += 1
            else:
                sim_intervals.extend(days_intervals)
                sim_ws.extend(w_interv_intervals_schl_open_set1)
                days_2 += 1
                if days_2 < altern_period:
                    days_1 = 0
                    days_2 = 0
        

    return sim_intervals, sim_ws
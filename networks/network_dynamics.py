from jax.dtypes import dtype
import jax.numpy as np
import numpy as np2
from jax import random
from networks import create_networks

def morning_set(Graphs_matrix, hh_occupation=9.0):

    # load networks
    matrix_household = Graphs_matrix[0]
    hh_row = np.asarray(np2.asarray(matrix_household[0]))
    hh_col = np.asarray(np2.asarray(matrix_household[1]))
    hh_data = np.asarray(np2.asarray(matrix_household[2]))

    matrix_school = Graphs_matrix[1]
    schl_row = np.asarray(np2.asarray(matrix_school[0]))
    schl_col = np.asarray(np2.asarray(matrix_school[1]))
    schl_data = np.asarray(np2.asarray(matrix_school[2]))

    matrix_work = Graphs_matrix[2]
    work_row = np.asarray(np2.asarray(matrix_work[0]))
    work_col = np.asarray(np2.asarray(matrix_work[1]))
    work_data = np.asarray(np2.asarray(matrix_work[2]))

    matrix_community = Graphs_matrix[3]
    comm_row = np.asarray(np2.asarray(matrix_community[0]))
    comm_col = np.asarray(np2.asarray(matrix_community[1]))
    comm_data = np.asarray(np2.asarray(matrix_community[2]))

    # turn off school and work layers
    schl_data_set = 0*schl_data
    work_data_set = 0*work_data

    # turn on portions of households and community
    comm_occupation = 1 - hh_occupation

    length = int(hh_data.shape[0]/2)
    hh_data_select = np.repeat( random.bernoulli(random.PRNGKey(0),p=(hh_occupation),
                                              shape=(length,)), 2) 
    hh_data_set = hh_data_select*hh_data

    length = int(comm_data.shape[0]/2)
    comm_data_select = np.repeat( random.bernoulli(random.PRNGKey(0),p=(comm_occupation),
                                              shape=(length,)), 2) 
    comm_data_set = comm_data_select*comm_data

    # create conections
    args_ps = (hh_data_set,schl_data_set,work_data_set,comm_data_set)
    ps = np.concatenate(args_ps)
    args_rows = (hh_row,schl_row,work_row,comm_row)
    rows = np.concatenate(args_rows)
    args_cols = (hh_col,schl_col,work_col,comm_col)

    # load graphs data
    hh_row,     hh_col,   hh_data = Graphs_matrix[0]
    schl_row, schl_col, schl_data = Graphs_matrix[1]
    work_row, work_col, work_data = Graphs_matrix[2]
    comm_row, comm_col, comm_data = Graphs_matrix[3]

    cols = np.concatenate(args_cols)

    w = [rows.astype(np.int32),cols.astype(np.int32),ps]

    return w


def evening_set(Graphs_matrix):

    # load networks
    matrix_household = Graphs_matrix[0]
    hh_row = np.asarray(np2.asarray(matrix_household[0]))
    hh_col = np.asarray(np2.asarray(matrix_household[1]))
    hh_data = np.asarray(np2.asarray(matrix_household[2]))

    matrix_school = Graphs_matrix[1]
    schl_row = np.asarray(np2.asarray(matrix_school[0]))
    schl_col = np.asarray(np2.asarray(matrix_school[1]))
    schl_data = np.asarray(np2.asarray(matrix_school[2]))

    matrix_work = Graphs_matrix[2]
    work_row = np.asarray(np2.asarray(matrix_work[0]))
    work_col = np.asarray(np2.asarray(matrix_work[1]))
    work_data = np.asarray(np2.asarray(matrix_work[2]))

    matrix_community = Graphs_matrix[3]
    comm_row = np.asarray(np2.asarray(matrix_community[0]))
    comm_col = np.asarray(np2.asarray(matrix_community[1]))
    comm_data = np.asarray(np2.asarray(matrix_community[2]))

    # turn off houses and community layers
    hh_data_set = 0*hh_data
    comm_data_set = 0*comm_data

    # create conections
    args_ps = (hh_data_set,schl_data,work_data,comm_data_set)
    ps = np.concatenate(args_ps)
    args_rows = (hh_row,schl_row,work_row,comm_row)
    rows = np.concatenate(args_rows)
    args_cols = (hh_col,schl_col,work_col,comm_col)
    cols = np.concatenate(args_cols)

    w = [rows.astype(np.int32),cols.astype(np.int32),ps]

    return w

def night_set(Graphs_matrix,hh_occupation=7.0):

    # load networks
    matrix_household = Graphs_matrix[0]
    hh_row = np.asarray(np2.asarray(matrix_household[0]))
    hh_col = np.asarray(np2.asarray(matrix_household[1]))
    hh_data = np.asarray(np2.asarray(matrix_household[2]))

    matrix_school = Graphs_matrix[1]
    schl_row = np.asarray(np2.asarray(matrix_school[0]))
    schl_col = np.asarray(np2.asarray(matrix_school[1]))
    schl_data = np.asarray(np2.asarray(matrix_school[2]))

    matrix_work = Graphs_matrix[2]
    work_row = np.asarray(np2.asarray(matrix_work[0]))
    work_col = np.asarray(np2.asarray(matrix_work[1]))
    work_data = np.asarray(np2.asarray(matrix_work[2]))

    matrix_community = Graphs_matrix[3]
    comm_row = np.asarray(np2.asarray(matrix_community[0]))
    comm_col = np.asarray(np2.asarray(matrix_community[1]))
    comm_data = np.asarray(np2.asarray(matrix_community[2]))

    # turn off school and work layers
    schl_data_set = 0*schl_data
    work_data_set = 0*work_data

    # turn on portions of households and community
    comm_occupation = 1 - hh_occupation

    length = int(hh_data.shape[0]/2)
    hh_data_select = np.repeat( random.bernoulli(random.PRNGKey(0),p=(hh_occupation),
                                              shape=(length,)), 2) 
    hh_data_set = hh_data_select*hh_data

    length = int(comm_data.shape[0]/2)
    comm_data_select = np.repeat( random.bernoulli(random.PRNGKey(0),p=(comm_occupation),
                                              shape=(length,)), 2) 
    comm_data_set = comm_data_select*comm_data

    # create conections
    args_ps = (hh_data_set,schl_data_set,work_data_set,comm_data_set)
    ps = np.concatenate(args_ps)
    args_rows = (hh_row,schl_row,work_row,comm_row)
    rows = np.concatenate(args_rows)
    args_cols = (hh_col,schl_col,work_col,comm_col)
    cols = np.concatenate(args_cols)

    w = [rows.astype(np.int32),cols.astype(np.int32),ps]

    return w


def create_day_dynamics(Graphs_matrix,Tmax,total_steps,partitions=[8,8,8]):
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

    m_w = morning_set(Graphs_matrix)
    e_w = evening_set(Graphs_matrix)
    n_w = night_set(Graphs_matrix)
    w_intervals = [m_w,e_w,n_w]

    sim_intervals = []  # iterations per network set w
    sim_ws        = []  # networks per iteration
    for d in range(Tmax):
        sim_intervals.extend(days_intervals)
        sim_ws.extend(w_intervals)

    return sim_intervals, sim_ws
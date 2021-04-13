""" Functions to create the various layers of the network """

import sys
sys.path.append('../')

from models import aerosol_transmission as at

import numpy as np2
import networkx as nx
import math
import itertools
import scipy.stats as ss
from collections import Counter

def calculate_individual_degree(rows):
    node_degree = dict(Counter(rows).items()) # Node degree
    return node_degree

def get_age_class(node_age):
    node_class = None
    if node_age < 4:
        node_class = 'schools'
    elif node_age >= 4 and node_age < 13:
        node_class = 'adults'
    else:
        node_class = 'elders'
    return node_class

def set_infection_prob(edges, ages, df_ages_params, delta_t, calculate_individual_degree=calculate_individual_degree):
    """ Set individual infection probability depending on the connection of an individiual.
    @param graph : Graph repesenting population struture
    @type : nx undirected graph
    @param R0 :  Basic reproductive number set for all the system
    @type : float
    @param duration : Duration of infectious period (days)
    @type : int 
    @param delta_t : Time stamp
    @type : float
    @return : Adjacency matrix in sparse format [rows, cols, data]
    @type : list of lists
    """
    rows = edges[0]
    cols = edges[1]
    degrees = calculate_individual_degree(rows)
    # Calculate infection probability
    ps = []
    for row_n in rows:
        # Get node degree
        deg_n = degrees[row_n] + 1
        # Get node age and class
        age_class_n = get_age_class(ages[row_n])
        # Get node params params depending in class
        R0_n = df_ages_params.loc[df_ages_params['layer']==age_class_n,'R0']
        duration_n = df_ages_params.loc[df_ages_params['layer']==age_class_n,'RecPeriod']
        # Calculate infection probability
        if deg_n == 1:
            prob_inf = 1e-6
        else:
            prob_inf = (R0_n/((deg_n-1)*duration_n))*delta_t
        ps.append(prob_inf)

    w = [rows, cols, ps]    # Arrange in list 

    return w

# def set_infection_prob_schools(lenght_room,width_room,height_room,ventilation_out,inhalation_mask_eff,exhalation_mask_eff,
#         fraction_people_masks,duration_event_h,edges, ages, df_ages_params, delta_t, calculate_individual_degree=calculate_individual_degree):
#     """ Set individual infection probability depending on the connection of an individiual.
#     @param graph : Graph repesenting population struture
#     @type : nx undirected graph
#     @param R0 :  Basic reproductive number set for all the system
#     @type : float
#     @param duration : Duration of infectious period (days)
#     @type : int 
#     @param delta_t : Time stamp
#     @type : float
#     @return : Adjacency matrix in sparse format [rows, cols, data]
#     @type : list of lists
#     """
#     rows = edges[0]
#     cols = edges[1]
#     degrees = calculate_individual_degree(rows)
#     # Calculate infection probability
#     ps = []
#     for row_n in rows:
#         # Get node degree
#         deg_n = degrees[row_n] + 1
#         # Get node age and class
#         age_class_n = get_age_class(ages[row_n])
#         # Get node params params depending in class
#         R0_n = df_ages_params.loc[df_ages_params['layer']==age_class_n,'R0']
#         duration_n = df_ages_params.loc[df_ages_params['layer']==age_class_n,'RecPeriod']
#         # Calculate infection probability
#         if deg_n == 1:
#             prob_inf = 1e-6
#         else:
#             prob_inf = (R0_n/((deg_n-1)*duration_n))*delta_t
#         ps.append(prob_inf)

#     w = [rows, cols, ps]    # Arrange in list 

#     return w

def set_infection_prob_schools(lenght_room,width_room,height_room,ventilation_out,inhalation_mask_eff,exhalation_mask_eff,
        fraction_people_masks,duration_event_h,edges, ages, df_ages_params, delta_t, calculate_individual_degree=calculate_individual_degree):
    """ Set individual infection probability depending on the connection of an individiual.
    @param graph : Graph repesenting population struture
    @type : nx undirected graph
    @param R0 :  Basic reproductive number set for all the system
    @type : float
    @param duration : Duration of infectious period (days)
    @type : int 
    @param delta_t : Time stamp
    @type : float
    @return : Adjacency matrix in sparse format [rows, cols, data]
    @type : list of lists
    """
    rows = edges[0]
    cols = edges[1]
    degrees = calculate_individual_degree(rows)
    # Calculate infection probability
    ps = []
    for row_n in rows:
        # Get node degree
        deg_n = degrees[row_n] + 1
        # Get node age and class
        age_class_n = get_age_class(ages[row_n])
        # Get node params params depending in class
        R0_n = df_ages_params.loc[df_ages_params['layer']==age_class_n,'R0']
        duration_n = df_ages_params.loc[df_ages_params['layer']==age_class_n,'RecPeriod']
        # Get aerosol transmission probability from being on a class
        aerosol_prob = at.infection_probability(lenght_room,width_room,height_room,ventilation_out,inhalation_mask_eff,
                                    exhalation_mask_eff,fraction_people_masks,duration_event_h)
        # Calculate infection probability
        if deg_n == 1:
            prob_inf = 1e-6 * 1
        else:
            prob_inf = (R0_n/((deg_n-1)*duration_n)) * aerosol_prob * delta_t * 100
        ps.append(prob_inf)
    w = [rows, cols, ps]    # Arrange in list 

    return w


def create_fully_connected(dist_groups, ages, indices, df_ages_params, delta_t):
    """ Divide the subset of the total population as given by the indices into fully connected groups 
    depending upon their distribution of sizes.
    @param dist_groups : Sizes of the groups in the population
    @type : list or 1D array
    @param indices : Indices of the subset of the population to be grouped together
    @type : list or 1D array
    @param R0 :  Basic reproductive number set for all the system
    @type : float
    @param duration : Duration of infectious period (days)
    @type : int 
    @param delta_t : Time stamp
    @type : float
    @return : Adjacency matrix in sparse format [rows, cols, data]
    @type : list of lists
    """
    rows = []
    cols = []
    current_indx = 0
    for size in dist_groups:
        group = indices[int(current_indx):int(current_indx+size)]
        current_indx += size
        comb = list(itertools.combinations(group,2))
        for i,j in comb:
            rows.extend([i,j])
            cols.extend([j,i])

    edges = [rows, cols]

    w = set_infection_prob(edges,ages,df_ages_params,delta_t)

    return w
    
    
def create_external_corr(pop,pop_subset,degree_dist,n,r,indx_list,correlation_group,ages,df_ages_params,delta_t):
    """ Create correlated external connections for either the whole population or a subset
    @param pop : Total population size
    @type : int
    @param pop_subset : Subset of the population involved in these external layers
    @type : int
    @param degree_dist : Degree distribution for this layer
    @type : list or 1D array
    @param n : Number of equal sized quantiles the correlated connections are divided into
    @type : int
    @param r : Amount of positive correlation between members of the same quantile
    @type : float
    @param indx_list : Array of indices of the individuals to be connected in the layer
    @type : list or 1D array
    @param correlation_group : Array of traits that are used to preferentially connect individuals
    @type : 1D array
    @param graph : Graph repesenting population struture
    @type : nx undirected graph
    @param R0 :  Basic reproductive number set for all the system
    @type : float
    @param duration : Duration of infectious period (days)
    @type : int 
    @param delta_t : Time stamp
    @type : float
    @return : Sparse adjacency matrix 
    @type : List of lists [rows, cols, data]
    """
    # Assign random and correlated stubs for each individual
    correlation = []
    np2.random.seed(789)
    for i in range(pop_subset):
        correlation.append(np2.random.binomial(1, r, size = degree_dist[i]))
    # Create external stubs that are randomly connected and the ones that are correlated for age groups
    rows = []
    cols = []
    zero_stubs = []
    one_stubs = {}
    
    for i in range(pop_subset):
        ones = np2.count_nonzero(correlation[i])
        zeros = degree_dist[i] - ones
        zero_stubs.extend([indx_list[i] for j in range(zeros)])
        if ones != 0:
            one_stubs[(indx_list[i],ones)] = correlation_group[i]
      
    # Attach the random stubs        
    zero_pairs = np2.random.choice(zero_stubs, size = (int(len(zero_stubs)/2),2), replace = False)
    for pairs in range(len(zero_pairs)):
        i = zero_pairs[pairs][0]
        j = zero_pairs[pairs][1] 
        rows.extend([i,j])    
        cols.extend([j,i])

    if r > 0:
        # Order correlated stubs according to trait to be correlated
        ordered_ones = sorted(one_stubs, key=one_stubs.__getitem__)
        sorted_ones = []
        for pairs in range(len(ordered_ones)):
            index = ordered_ones[pairs][0]
            sorted_ones.extend([index for i in range(ordered_ones[pairs][1])])
    
        # Divide into n_school number of equal sized quantiles
        n_q = math.ceil(len(sorted_ones)/n)
        n_quantiles = [sorted_ones[i:i + n_q] for i in range(0, len(sorted_ones), n_q)]

        # Attach the correlated nodes
        for quantile in range(len(n_quantiles)):
            one_pairs = np2.random.choice(n_quantiles[quantile], size = (int(len(n_quantiles[quantile])/2),2), replace = False)
            for pairs in range(len(one_pairs)):
                i = one_pairs[pairs][0]
                j = one_pairs[pairs][1]
                rows.extend([i,j])
                cols.extend([j,i])

    edges = [rows, cols]

    w = set_infection_prob(edges,ages,df_ages_params,delta_t)

    return w


def create_external_corr_schools(pop,pop_subset,degree_dist,n,r,indx_list,correlation_group,ages,df_ages_params,delta_t,
        lenght_room,width_room,height_room,ventilation_out,inhalation_mask_eff,exhalation_mask_eff,fraction_people_masks,duration_event_h):
    """ Create correlated external connections for either the whole population or a subset
    @param pop : Total population size
    @type : int
    @param pop_subset : Subset of the population involved in these external layers
    @type : int
    @param degree_dist : Degree distribution for this layer
    @type : list or 1D array
    @param n : Number of equal sized quantiles the correlated connections are divided into
    @type : int
    @param r : Amount of positive correlation between members of the same quantile
    @type : float
    @param indx_list : Array of indices of the individuals to be connected in the layer
    @type : list or 1D array
    @param correlation_group : Array of traits that are used to preferentially connect individuals
    @type : 1D array
    @param graph : Graph repesenting population struture
    @type : nx undirected graph
    @param R0 :  Basic reproductive number set for all the system
    @type : float
    @param duration : Duration of infectious period (days)
    @type : int 
    @param delta_t : Time stamp
    @type : float
    @return : Sparse adjacency matrix 
    @type : List of lists [rows, cols, data]
    """
    # Assign random and correlated stubs for each individual
    correlation = []
    np2.random.seed(789)
    for i in range(pop_subset):
        correlation.append(np2.random.binomial(1, r, size = degree_dist[i]))
    # Create external stubs that are randomly connected and the ones that are correlated for age groups
    rows = []
    cols = []
    zero_stubs = []
    one_stubs = {}
    
    for i in range(pop_subset):
        ones = np2.count_nonzero(correlation[i])
        zeros = degree_dist[i] - ones
        zero_stubs.extend([indx_list[i] for j in range(zeros)])
        if ones != 0:
            one_stubs[(indx_list[i],ones)] = correlation_group[i]
      
    # Attach the random stubs        
    zero_pairs = np2.random.choice(zero_stubs, size = (int(len(zero_stubs)/2),2), replace = False)
    for pairs in range(len(zero_pairs)):
        i = zero_pairs[pairs][0]
        j = zero_pairs[pairs][1] 
        rows.extend([i,j])    
        cols.extend([j,i])

    if r > 0:
        # Order correlated stubs according to trait to be correlated
        ordered_ones = sorted(one_stubs, key=one_stubs.__getitem__)
        sorted_ones = []
        for pairs in range(len(ordered_ones)):
            index = ordered_ones[pairs][0]
            sorted_ones.extend([index for i in range(ordered_ones[pairs][1])])
    
        # Divide into n_school number of equal sized quantiles
        n_q = math.ceil(len(sorted_ones)/n)
        n_quantiles = [sorted_ones[i:i + n_q] for i in range(0, len(sorted_ones), n_q)]

        # Attach the correlated nodes
        for quantile in range(len(n_quantiles)):
            one_pairs = np2.random.choice(n_quantiles[quantile], size = (int(len(n_quantiles[quantile])/2),2), replace = False)
            for pairs in range(len(one_pairs)):
                i = one_pairs[pairs][0]
                j = one_pairs[pairs][1]
                rows.extend([i,j])
                cols.extend([j,i])

    edges = [rows, cols]

    w = set_infection_prob_schools(lenght_room,width_room,height_room,ventilation_out,inhalation_mask_eff,
                exhalation_mask_eff,fraction_people_masks,duration_event_h,edges,ages,df_ages_params,delta_t)

    return w


    
def create_friend_groups(para,age_grp_size,indices):
    """ Create age dependent distributions of sizes of friend groups and assign individuals to them
    @param para : List of parameters for the negative binomial distribution [n,p]
    @type : list
    @param age_grp_size : Number of individuals in an age group
    @type : int
    @param indices : Indices of the subset of the population to be grouped together
    @type : list or 1D array
    @return : Sparse adjacency matrix per age group
    @type : List of lists [rows, cols, data]
    
    """
    group_sizes = []
    pop_group = 0
    n = para[0]
    p = para[1]
    
    np2.random.seed(789)
    while pop_group <= age_grp_size:
        size = np2.random.negative_binomial(n,p, size=1)
        group_sizes.append(size)
        pop_group += size

    group_sizes[-1] -= pop_group-age_grp_size
    sparse_matrix = create_fully_connected(group_sizes,indices)
    return sparse_matrix


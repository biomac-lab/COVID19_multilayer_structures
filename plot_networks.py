import networkx as nx
import pandas as pd
import jax.numpy as np
import numpy as np2
import os
import itertools
import matplotlib.pyplot as plt
from tqdm import tqdm

config_data = pd.read_csv('config.csv', sep=',', header=None, index_col=0)
figures_path = config_data.loc['figures_dir'][1]
multilayers_path = config_data.loc['multilayers_dir'][1]
ages_data_path = config_data.loc['bogota_age_data_dir'][1]
houses_data_path = config_data.loc['bogota_houses_data_dir'][1]

# from networks import networks
from networks import create_networks

import argparse
parser = argparse.ArgumentParser(description='Networks visualization.')
parser.add_argument('--population', default=1000, type=int,
                    help='Speficy the number of individials')
parser.add_argument('--save', default=False, type=bool,
                    help='Speficy if you want to save the networks')
parser.add_argument('--plot', default=True, type=bool,
                    help='Speficy if you want to save the figure')

parser.add_argument('--schools_mean', default=9.4, type=float,
                    help='Schools degree distribution (mean)')
parser.add_argument('--schools_std', default=1.8, type=float,
                    help='Schools degree distribution (standard deviation)')
parser.add_argument('--schools_size', default=35, type=float,
                    help='Number of students per classroom')
parser.add_argument('--schools_r', default=1, type=float,
                    help='Correlation in schools layer')

parser.add_argument('--work_mean', default=14.4/3, type=float,
                    help='Work degree distribution (mean)')
parser.add_argument('--work_std', default=6.2/3, type=float,
                    help='Work degree distribution (standard deviation)')
parser.add_argument('--work_size', default=10, type=float,
                    help='Approximation of a work place size')
parser.add_argument('--work_r', default=1, type=float,
                    help='Correlation in work layer')

parser.add_argument('--community_mean', default=4.3/2, type=float,
                    help='Community degree distribution (mean)')
parser.add_argument('--community_std', default=1.9/2, type=float,
                    help='Community degree distribution (standard deviation)')
parser.add_argument('--community_n', default=1, type=float,
                    help='Number of community')
parser.add_argument('--community_r', default=0, type=float,
                    help='Correlation in community layer')

parser.add_argument('--R0', default=3, type=float,
                    help='Fixed basic reproduction number') 
parser.add_argument('--MILDINF_DURATION', default=6, type=int,
                    help='Duration of mild infection, days')   
parser.add_argument('--delta_t', default=0.1, type=float,
                    help='Time stamp')  
                                         
args = parser.parse_args()


number_nodes = args.population
pop = number_nodes


### Get age distribution
ages_data_BOG = pd.read_csv(ages_data_path, encoding= 'unicode_escape', delimiter=';')
total_pop_BOG = int(ages_data_BOG['Total.3'][17].replace('.',''))

# Ages 0-4
very_young_ = [int(ages_data_BOG['Total.3'][0].replace('.',''))]
very_young = sum(very_young_)/total_pop_BOG

# Ages 5-19
school_ = [int(ages_data_BOG['Total.3'][i].replace('.','')) for i in range(1,3+1)]
school = sum(school_)/total_pop_BOG

# Ages 19-24
university_ = int(ages_data_BOG['Total.3'][4].replace('.',''))
university = int(ages_data_BOG['Total.3'][4].replace('.',''))/total_pop_BOG

# Ages 24-64
work_ = [int(ages_data_BOG['Total.3'][i].replace('.','')) for i in range(5,12+1)]
work = sum(work_)/total_pop_BOG

# Ages 65+
elderly_ = [int(ages_data_BOG['Total.3'][i].replace('.','')) for i in range(13,16+1)]
elderly = sum(elderly_)/total_pop_BOG

# Community ages
community_ = very_young_ + school_ + [university_] + work_ + elderly_
community = sum(community_)/total_pop_BOG


### Get household size distribution from 2018 census data
census_data_BOG = pd.read_csv(houses_data_path)
one_house   = np2.sum(census_data_BOG['HA_TOT_PER'] == 1.0)
two_house   = np2.sum(census_data_BOG['HA_TOT_PER'] == 2.0)
three_house = np2.sum(census_data_BOG['HA_TOT_PER'] == 3.0)
four_house  = np2.sum(census_data_BOG['HA_TOT_PER'] == 4.0)
five_house  = np2.sum(census_data_BOG['HA_TOT_PER'] == 5.0)
six_house   = np2.sum(census_data_BOG['HA_TOT_PER'] == 6.0)
seven_house = np2.sum(census_data_BOG['HA_TOT_PER'] == 7.0)
total_house = one_house + two_house + three_house + four_house + five_house + six_house + seven_house 

house_size_dist = np2.array([one_house,two_house,three_house,four_house,five_house,six_house,seven_house])/total_house

# House-hold sizes
household_sizes = []

household_sizes.extend(np2.random.choice(np.arange(1,8,1),p=house_size_dist,size=int(pop/3))) # This division is just to make the code faster
pop_house = sum(household_sizes)

while pop_house <= pop:
    size = np2.random.choice(np.arange(1,8,1),p=house_size_dist,size=1)
    household_sizes.extend(size)
    pop_house += size[0]

household_sizes[-1] -= pop_house-pop

# Mean of household degree dist 
mean_household = sum((np2.asarray(household_sizes)-1)*np2.asarray(household_sizes))/pop

# Keeping track of the household indx for each individual
house_indices = np2.repeat(np2.arange(0,len(household_sizes),1), household_sizes)

# Keeping track of the household size for each individual
track_house_size = np2.repeat(household_sizes, household_sizes)

# Keep track of the 5 yr age groups for each individual labelled from 0-16
age_tracker_all = np2.zeros(pop)

####### Community 
# Degree dist. mean and std div obtained by Prem et al data, scaled by 1/2.5 in order to ensure that community+friends+school = community data in Prem et al
mean, std = args.community_mean, args.community_std
p = 1-(std**2/mean)
n_binom = mean/p
community_degree = np2.random.binomial(n_binom, p, size = pop)

# No correlation between contacts
n_community = args.community_n
r_community = args.community_r

# Split the age group of old population according to the population seen in the data
prob = []
for i in range(0,len(community_)):
    prob.append(community_[i]/sum(community_))
age_group_community = np2.random.choice(np2.arange(0,len(community_),1),size=pop,p=prob,replace=True)

community_indx = np2.arange(0,pop,1)
for i in range(pop):
    age_tracker_all[community_indx[i]] = age_group_community[i]

# Keep track of the 5 yr age groups for each individual labelled from 0-16


###############################
##### Degree distribution #####

# Frac of population that is school going, working, preschool or elderly
dist_of_pop = [school,work,very_young+university+elderly]

# Classifying each person
classify_pop = np2.random.choice(['schools','work','other'], size=pop, p=dist_of_pop)

# Number of individuals in each group
state, counts = np2.unique(classify_pop, return_counts=True)
dict_of_counts = dict(zip(state,counts))
school_going = dict_of_counts['schools']
working = dict_of_counts['work']
other = dict_of_counts['other']

# Indices of individuals in each group
school_indx = np2.where(classify_pop=='schools')[0]
work_indx = np2.where(classify_pop=='work')[0]
other_indx = np2.where(classify_pop=='other')[0]

age_tracker = np2.zeros(pop)

####### schools
mean, std = args.schools_mean, args.schools_std
p = 1-(std**2/mean)
n_binom = mean/p
schools_degree = np2.random.binomial(n_binom, p, size = school_going)
n_school = school_going/args.schools_size
r_school = args.schools_r

school_clroom = np2.random.choice(np.arange(0,n_school+1,1),size=school_going)

# Assign ages to the school going population acc. to their proportion from the census data
prob = []
for i in range(0,len(school_)):
    prob.append(school_[i]/sum(school_))
age_group_school = np2.random.choice([1,2,3],size=school_going,p=prob,replace=True)

for i in range(school_going):
    age_tracker[school_indx[i]] = age_group_school[i]


####### Work 
# Degree dist., the mean and std div have been taken from the Potter et al data. The factor of 1/3 is used to correspond to daily values and is chosen to match with the work contact survey data
mean, std = args.work_mean, args.work_std
p = 1-(std**2/mean)
n_binom = mean/p
work_degree = np2.random.binomial(n_binom, p, size = working)

# Assuming that on average the size of a work place is ~ 10 people and the correlation is 
# chosen such that the clustering coeff is high as the network in Potter et al had a pretty high value
work_place_size = args.work_size
n_work = working/work_place_size
r_work = args.work_r

# Assign each working individual a 'work place'
job_place = np2.random.choice(np.arange(0,n_work+1,1),size=working)

# Split the age group of working population according to the population seen in the data
p = []
for i in range(0,len(work_)):
    p.append(work_[i]/sum(work_))
age_group_work = np2.random.choice(np.arange(0,len(work_),1),size=working,p=p,replace=True)

for i in range(working):
    age_tracker[work_indx[i]] = age_group_work[i]


## Households
matrix_household = create_networks.create_fully_connected(household_sizes,np2.arange(0,pop,1),args.R0,args.MILDINF_DURATION,args.delta_t)
# matrix_household = networks.create_fully_connected(household_sizes,np2.arange(0,pop,1))

# Get row, col, data information from the sparse matrices
# Converting into DeviceArrays to run faster with jax. Not sure why the lists have to be first converted to usual numpy arrays though
matrix_household_row = np.asarray(np2.asarray(matrix_household[0]))
matrix_household_col = np.asarray(np2.asarray(matrix_household[1]))
matrix_household_data = np.asarray(np2.asarray(matrix_household[2]))

## School

matrix_school = create_networks.create_external_corr(pop,school_going,schools_degree,n_school,r_school,school_indx,school_clroom,args.R0,args.MILDINF_DURATION,args.delta_t)

matrix_school_row = np.asarray(np2.asarray(matrix_school[0]))
matrix_school_col = np.asarray(np2.asarray(matrix_school[1]))
matrix_school_data = np.asarray(np2.asarray(matrix_school[2]))

## Work

matrix_work = create_networks.create_external_corr(pop,working,work_degree,n_work,r_work,work_indx,job_place,args.R0,args.MILDINF_DURATION,args.delta_t)
# matrix_work = networks.create_external_corr(pop,working,work_degree,n_work,r_work,work_indx,job_place)

matrix_work_row = np.asarray(np2.asarray(matrix_work[0]))
matrix_work_col = np.asarray(np2.asarray(matrix_work[1]))
matrix_work_data = np.asarray(np2.asarray(matrix_work[2]))

## Community

matrix_community = create_networks.create_external_corr(pop,pop,community_degree,n_community,r_community,community_indx,age_group_community,args.R0,args.MILDINF_DURATION,args.delta_t)
# matrix_community = create_networks.create_external_corr(pop,pop,community_degree,n_community,r_community,community_indx,age_group_community)

matrix_community_row = np.asarray(np2.asarray(matrix_community[0]))
matrix_community_col = np.asarray(np2.asarray(matrix_community[1]))
matrix_community_data = np.asarray(np2.asarray(matrix_community[2]))

# Mean degree of household and external layers
mean_house = sum(matrix_household_data)/pop
mean_school = sum(matrix_school_data)/school_going
mean_work = sum(matrix_work_data)/working
mean_community = sum(matrix_community_data)/pop

print("Mean degree household = %0.2f"%mean_house)
print("Mean degree school = %0.2f"%mean_school)
print("Mean degree work = %0.2f"%mean_work)
print("Mean degree community = %0.2f"%mean_community)

# Combine the data arrays later depending upon the weights needed for the simulations

args_rows = (matrix_household_row, matrix_school_row, matrix_work_row, matrix_community_row)
args_cols = (matrix_household_col, matrix_school_col, matrix_work_col, matrix_community_col)
rows = np.concatenate(args_rows)
cols = np.concatenate(args_cols)

# Get Edges as tuples
house_edge_list = []
for i in range(matrix_household_row.size):
    edge_i = (np2.array(matrix_household_row)[i], np2.array(matrix_household_col)[i])
    house_edge_list.append(edge_i)

school_edge_list = []
for i in range(matrix_school_row.size):
    edge_i = (np2.array(matrix_school_row)[i], np2.array(matrix_school_col)[i])
    school_edge_list.append(edge_i) 

work_edge_list = []
for i in range(matrix_work_row.size):
    edge_i = (np2.array(matrix_work_row)[i], np2.array(matrix_work_col)[i])
    work_edge_list.append(edge_i)  

community_edge_list = []
for i in range(matrix_community_row.size):
    edge_i = (np2.array(matrix_community_row)[i], np2.array(matrix_community_col)[i])
    community_edge_list.append(edge_i) 

# Create graphs
print('Creating graphs')
household_G = nx.Graph()
household_G.add_edges_from(house_edge_list)
household_G = nx.DiGraph.to_undirected(household_G)


school_G = nx.Graph()
school_G.add_edges_from(school_edge_list)
school_G = nx.DiGraph.to_undirected(school_G)

work_G = nx.Graph()
work_G.add_edges_from(work_edge_list)
work_G = nx.DiGraph.to_undirected(work_G)

community_G = nx.Graph()
community_G.add_edges_from(community_edge_list)
community_G = nx.DiGraph.to_undirected(community_G)

multilayer4_G = [household_G,school_G,work_G,community_G]
multilayer4_l = ['household_G','school_G','work_G','community_G']

if args.save:
    if not os.path.isdir( os.path.join(multilayers_path, str(number_nodes)) ):
        os.makedirs( os.path.join(multilayers_path, str(number_nodes)) )
    path_save = os.path.join(multilayers_path, str(number_nodes))
    # Save pickle
    print('Saving networks')
    for i in tqdm(range(0,len(multilayer4_G))):
        nx.write_gpickle(multilayer4_G[i], os.path.join(path_save,'{}_{}.pickle'.format(multilayer4_l[i],str(number_nodes))))

if args.plot:
    if not os.path.isdir( os.path.join(figures_path, str(number_nodes)) ):
        os.makedirs( os.path.join(figures_path, str(number_nodes)) )
    # Plot and save
    print('Creating figures')
    for i in tqdm(range(0,len(multilayer4_G))):
        plt.figure(figsize=(15,15))
        pos = nx.kamada_kawai_layout(multilayer4_G[i])
        nx.draw(G=multilayer4_G[i], pos=pos, 
            node_size=8,
            node_color= 'black',
            edge_color='gray',
            width=.5,
            edge_cmap=plt.cm.Blues, with_labels=False)
        plt.savefig(os.path.join(figures_path, str(pop)+'_{}.png'.format(multilayer4_l[i])), dpi=400, transparent=False, bbox_inches = 'tight', pad_inches = 0.1)

print('Done!')
#!
import jax.numpy as np
from jax import jit, random, vmap
from jax.ops import index_add, index_update, index
import matplotlib.pyplot as plt
import functools
import itertools
from scipy import optimize
from scipy.special import gamma
from tqdm import tqdm
import numpy as np2
import pandas as pd
import pickle
import os

from models import model

config_data = pd.read_csv('configlin.csv', sep=',', header=None, index_col=0)
figures_path = config_data.loc['figures_dir'][1]
results_path = config_data.loc['results_dir'][1]
ages_data_path = config_data.loc['bogota_age_data_dir'][1]
houses_data_path = config_data.loc['bogota_houses_data_dir'][1]

#from networks import networks
from networks import create_networks

import argparse
parser = argparse.ArgumentParser(description='Networks visualization.')

parser.add_argument('--population', default=1000, type=int,
                    help='Speficy the number of individials')

parser.add_argument('--Tmax', default=150, type=int,
                    help='Length of simulation (days)')
parser.add_argument('--delta_t', default=0.08, type=float,
                    help='Time steps')
parser.add_argument('--number_trials', default=10, type=int,
                    help='Number of iterations per step')

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
                                         
args = parser.parse_args()


number_nodes = args.population
pop = number_nodes


## Parameters
# Model parameter values

# Means
IncubPeriod=5  #Incubation period, days
DurMildInf=6 #Duration of mild infections, days
DurSevereInf=6 #Duration of hospitalization (severe infection), days
DurCritInf=8 #Time from ICU admission to death/recovery (critical infection), days

# Standard deviations
std_IncubPeriod=4  #Incubation period, days
std_DurMildInf=2 #Duration of mild infections, days
std_DurSevereInf=4.5 #Duration of hospitalization (severe infection), days
std_DurCritInf=6 #Time from ICU admission to death/recovery (critical infection), days

FracSevere=0.15 #Fraction of infections that are severe
FracCritical=0.05 #Fraction of infections that are critical
CFR=0.02 #Case fatality rate (fraction of infections resulting in death)
FracMild=1-FracSevere-FracCritical  #Fraction of infections that are mild

# Get gamma distribution parameters
mean_vec = np.array(
      [1., IncubPeriod, DurMildInf, DurSevereInf, DurCritInf, 1., 1.])
std_vec=np.array(
      [1., std_IncubPeriod, std_DurMildInf, std_DurSevereInf, std_DurCritInf, 1., 1.])
shape_vec=(mean_vec/std_vec)**2# This will contain shape values for each state
scale_vec=(std_vec**2)/mean_vec # This will contain scale values for each state

# Define transition probabilities

# Define probability of recovering (as opposed to progressing or dying) from each state
recovery_probabilities = np.array([0., 0., FracMild, FracSevere / (FracSevere + FracCritical), 1. - CFR / FracCritical, 0., 0.])

# Define relative infectivity of each state
infection_probabilities = np.array([0., 0., 1.0, 0., 0., 0., 0.])

def discrete_gamma(key, alpha, beta, shape=()):
  shape_ = shape
  if shape_ == ():
    try:
      shape_ = alpha.shape
    except:
      shape_ = ()
  return _discrete_gamma(key, alpha, beta, shape_)


@functools.partial(jit, static_argnums=(3,))
def _discrete_gamma(key, alpha, beta, shape=()):
  samples = np.round(random.gamma(key, alpha, shape=shape) / beta)
  return samples.astype(np.int32)


@jit
def state_length_sampler(key, new_state):
  """Duration in transitional state. Must be at least 1 time unit."""
  alphas = shape_vec[new_state]
  betas = delta_t/scale_vec[new_state]
  key, subkey = random.split(key)
  lengths = 1 + discrete_gamma(subkey, alphas, betas)    # Time must be at least 1.
  return key, lengths * model.is_transitional(new_state)    # Makes sure non-transitional states are returning 0.



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

print('Creating graphs...')

## Households
matrix_household = create_networks.create_fully_connected(household_sizes,np2.arange(0,pop,1),args.R0,args.MILDINF_DURATION,args.delta_t)

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

matrix_work_row = np.asarray(np2.asarray(matrix_work[0]))
matrix_work_col = np.asarray(np2.asarray(matrix_work[1]))
matrix_work_data = np.asarray(np2.asarray(matrix_work[2]))

## Community
matrix_community = create_networks.create_external_corr(pop,pop,community_degree,n_community,r_community,np2.arange(0,pop,1),age_group_community,args.R0,args.MILDINF_DURATION,args.delta_t)

matrix_community_row = np.asarray(np2.asarray(matrix_community[0]))
matrix_community_col = np.asarray(np2.asarray(matrix_community[1]))
matrix_community_data = np.asarray(np2.asarray(matrix_community[2]))

# Save graphs matrix
multilayer_matrix = [matrix_household,matrix_school,matrix_work,matrix_community]

# Time paramas
Tmax = args.Tmax
days_intervals = [1] * Tmax
delta_t = args.delta_t
tvec = np.arange(0,Tmax,delta_t)
step_intervals = [int(x/delta_t) for x in days_intervals]
total_steps = sum(step_intervals)

# Create dynamic
import networks.network_dynamics as nd

print('Creating dynamics...')

time_intervals, ws = nd.create_day_dynamics(multilayer_matrix,Tmax=Tmax,total_steps=total_steps)

# Bogota data
BOG_E = int(582085*(pop/total_pop_BOG))
BOG_R = int(pop*0.3)    # Assuming that 30% of population is already recovered
# BOG_R = int(520853*(pop/total_pop_BOG))
BOG_D = int(11787*(pop/total_pop_BOG))


####################### RUN
print('Simulating...')
soln=np.zeros((args.number_trials,total_steps,7))
soln_cum=np.zeros((args.number_trials,total_steps,7))

for key in tqdm(range(args.number_trials), total=args.number_trials):

  #Initial condition
  init_ind_E = random.uniform(random.PRNGKey(key), shape=(BOG_E,), maxval=pop).astype(np.int32)
  init_ind_R = random.uniform(random.PRNGKey(key), shape=(BOG_R,), maxval=pop).astype(np.int32)
  init_ind_D = random.uniform(random.PRNGKey(key), shape=(BOG_D,), maxval=pop).astype(np.int32)
  init_state = np.zeros(pop, dtype=np.int32)
  init_state = index_update(init_state,init_ind_E,np.ones(BOG_E, dtype=np.int32)*1) # E
  init_state = index_update(init_state,init_ind_D,np.ones(BOG_D, dtype=np.int32)*5) # D
  init_state = index_update(init_state,init_ind_R,np.ones(BOG_R, dtype=np.int32)*6) # R


  _, init_state_timer = state_length_sampler(random.PRNGKey(key), init_state)

  #Run simulation
  _, state, _, _, total_history = model.simulate_intervals(
    ws, time_intervals, state_length_sampler, infection_probabilities, 
    recovery_probabilities, init_state, init_state_timer, key = random.PRNGKey(key), epoch_len=1)
  
  history = np.array(total_history)[:, 0, :]  # This unpacks current state counts
  soln=index_add(soln,index[key,:, :],history)

  cumulative_history = np.array(total_history)[:, 1, :] 
  soln_cum=index_add(soln_cum,index[key,:, :],cumulative_history)

# Confidence intervals
loCI = 5
upCI = 95
soln_avg=np.average(soln,axis=0)
soln_loCI=np.percentile(soln,loCI,axis=0)
soln_upCI=np.percentile(soln,upCI,axis=0)

print('Saving results...')

# Save results

df_results_history = pd.DataFrame(columns=['tvec','S','E','I1','I2','I3','D','R'])
df_results_history['tvec']  = list(tvec[:soln.shape[1]])
df_results_history['S']     = list(history[:,0])
df_results_history['E']     = list(history[:,1])
df_results_history['I1']    = list(history[:,2])
df_results_history['I2']    = list(history[:,3])
df_results_history['I3']    = list(history[:,4])
df_results_history['D']     = list(history[:,5])
df_results_history['R']     = list(history[:,6])

df_results_mean = pd.DataFrame(columns=['tvec','S','E','I1','I2','I3','D','R'])
df_results_mean['tvec']  = list(tvec[:soln.shape[1]])
df_results_mean['S']     = list(soln_avg[:,0])
df_results_mean['E']     = list(soln_avg[:,1])
df_results_mean['I1']    = list(soln_avg[:,2])
df_results_mean['I2']    = list(soln_avg[:,3])
df_results_mean['I3']    = list(soln_avg[:,4])
df_results_mean['D']     = list(soln_avg[:,5])
df_results_mean['R']     = list(soln_avg[:,6])

df_results_loCI = pd.DataFrame(columns=['tvec','S','E','I1','I2','I3','D','R'])
df_results_loCI['tvec']  = list(tvec[:soln.shape[1]])
df_results_loCI['S']     = list(soln_loCI[:,0])
df_results_loCI['E']     = list(soln_loCI[:,1])
df_results_loCI['I1']    = list(soln_loCI[:,2])
df_results_loCI['I2']    = list(soln_loCI[:,3])
df_results_loCI['I3']    = list(soln_loCI[:,4])
df_results_loCI['D']     = list(soln_loCI[:,5])
df_results_loCI['R']     = list(soln_loCI[:,6])

df_results_upCI = pd.DataFrame(columns=['tvec','S','E','I1','I2','I3','D','R'])
df_results_upCI['tvec']  = list(tvec[:soln.shape[1]])
df_results_upCI['S']     = list(soln_upCI[:,0])
df_results_upCI['E']     = list(soln_upCI[:,1])
df_results_upCI['I1']    = list(soln_upCI[:,2])
df_results_upCI['I2']    = list(soln_upCI[:,3])
df_results_upCI['I3']    = list(soln_upCI[:,4])
df_results_upCI['D']     = list(soln_upCI[:,5])
df_results_upCI['R']     = list(soln_upCI[:,6])


if not os.path.isdir( os.path.join(results_path, str(number_nodes)) ):
        os.makedirs(os.path.join(results_path, str(number_nodes)))

path_save = os.path.join(results_path, str(number_nodes))

df_results_mean.to_csv(path_save+'/{}_history.csv'.format(str(number_nodes)), index=False)
df_results_mean.to_csv(path_save+'/{}_mean.csv'.format(str(number_nodes)), index=False)
df_results_loCI.to_csv(path_save+'/{}_loCI.csv'.format(str(number_nodes)), index=False)
df_results_upCI.to_csv(path_save+'/{}_upCI.csv'.format(str(number_nodes)), index=False)


# Save other statistics
per_day = int(1/delta_t)
soln_smooth=model.smooth_timecourse(soln,int(per_day/2)) # Smoothening over a day
model.get_peaks_iter(soln_smooth,tvec)


print('Done!')
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
params_data_path = config_data.loc['bogota_params_ages_data'][1]
ages_data_path = config_data.loc['bogota_age_data_dir'][1]
houses_data_path = config_data.loc['bogota_houses_data_dir'][1]
teachers_data_path = config_data.loc['bogota_teachers_data_dir'][1]

#from networks import networks
from networks import create_networks

import argparse
parser = argparse.ArgumentParser(description='Simulating interventions')

parser.add_argument('--population', default=1000, type=int,
                    help='Speficy the number of individials')
parser.add_argument('--intervention', default=0.6, type=float,
                    help='Intervention efficiancy')
parser.add_argument('--work_occupation', default=0.6, type=float,
                    help='Percentage of occupation at workplaces over intervention')
parser.add_argument('--school_occupation', default=0.35, type=float,
                    help='Percentage of occupation at classrooms over intervention')
parser.add_argument('--school_openings', default=20, type=int,
                    help='Day of the simulation where schools are open')
parser.add_argument('--school_alternancy', default=False, type=bool,
                    help='Percentage of occupation at classrooms over intervention')

parser.add_argument('--Tmax', default=180, type=int,
                    help='Length of simulation (days)')
parser.add_argument('--delta_t', default=0.08, type=float,
                    help='Time steps')
parser.add_argument('--number_trials', default=10, type=int,
                    help='Number of iterations per step')

parser.add_argument('--preschool_mean', default=9.4, type=float,
                    help='preschool degree distribution (mean)')
parser.add_argument('--preschool_std', default=1.8, type=float,
                    help='preschool degree distribution (standard deviation)')
parser.add_argument('--preschool_size', default=15, type=float,
                    help='Number of students per classroom')
parser.add_argument('--preschool_r', default=1, type=float,
                    help='Correlation in preschool layer')

parser.add_argument('--primary_mean', default=9.4, type=float,
                    help='primary degree distribution (mean)')
parser.add_argument('--primary_std', default=1.8, type=float,
                    help='primary degree distribution (standard deviation)')
parser.add_argument('--primary_size', default=35, type=float,
                    help='Number of students per classroom')
parser.add_argument('--primary_r', default=1, type=float,
                    help='Correlation in primary layer')

parser.add_argument('--highschool_mean', default=9.4, type=float,
                    help='highschool degree distribution (mean)')
parser.add_argument('--highschool_std', default=1.8, type=float,
                    help='highschool degree distribution (standard deviation)')
parser.add_argument('--highschool_size', default=35, type=float,
                    help='Number of students per classroom')
parser.add_argument('--highschool_r', default=1, type=float,
                    help='Correlation in highschool layer')

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


#--------------------------------------------------------------------------------------------------------------------------------

################################
########## Parameters ##########

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
recovery_probabilities = np.array([0., 0., Fcommunity_ = very_young_ + preschool_ + primary_ + highschool_ + university_ + work_ + elderly_
community = sum(community_)/total_pop_BOG
# Define relative infectivity of each state
infection_probabilities = np.array([0., 0., 1.0, 0., 0., 0., 0.])


#----------------------------------------------------------------------------------------------------------------------------------


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


#-----------------------------------------------------------------------------------------------------------------------------------

######################################
######## Teachers distribution #######

teachers_data_BOG = pd.read_csv(teachers_data_path, encoding= 'unicode_escape', delimiter=',')
total_teachers_BOG = int(teachers_data_BOG['Total'][1])

teachers_preschool_ = [int(teachers_data_BOG['Preescolar'][1])]
teachers_preschool = sum(teachers_preschool_)/total_teachers_BOG

teachers_primary_ = [int(teachers_data_BOG['Basica_primaria'][1])]
teachers_primary = sum(teachers_primary_)/total_teachers_BOG

teachers_highschool_ = [int(teachers_data_BOG['Basica_secundaria'][1])]
teachers_highschool = sum(teachers_highschool_)/total_teachers_BOG


#-----------------------------------------------------------------------------------------------------------------------------------

#################################
######## Age distribution #######

### Get age distribution
ages_data_BOG = pd.read_csv(ages_data_path, encoding= 'unicode_escape', delimiter=';')
total_pop_BOG = int(ages_data_BOG['Total.3'][17].replace('.',''))

# Ages 0-4 (0)
very_young_ = [int(ages_data_BOG['Total.3'][0].replace('.',''))]
very_young = sum(very_young_)/total_pop_BOG

# Ages 5-9 (1)
preschool_ = [int(ages_data_BOG['Total.3'][1].replace('.',''))]
preschool = sum(preschool_)/total_pop_BOG

# Ages 10-14 (2)
primary_ = [int(ages_data_BOG['Total.3'][2].replace('.',''))]
primary = sum(primary_)/total_pop_BOG

# Ages 15-19 (3)
highschool_ = [int(ages_data_BOG['Total.3'][3].replace('.',''))]
highschool = sum(highschool_)/total_pop_BOG

# Ages 20-24 (4)
university_ = [int(ages_data_BOG['Total.3'][4].replace('.',''))]
university = sum(university_)/total_pop_BOG

# Ages 25-64 (5,6,7,8,9,10,11,12)
work_ = [int(ages_data_BOG['Total.3'][i].replace('.','')) for i in range(5,12+1)]
work = sum(work_)/total_pop_BOG

# Ages 65+ (13,14,15,16)
elderly_ = [int(ages_data_BOG['Total.3'][i].replace('.','')) for i in range(13,16+1)]
elderly = sum(elderly_)/total_pop_BOG

# Community ages
community_ = very_young_ + preschool_ + primary_ + highschool_ + university_ + work_ + elderly_
community = sum(community_)/total_pop_BOG

# Adult classification
adults = np.arange(4,16+1,1)

#-----------------------------------------------------------------------------------------------------------------------------------

#################################
########### Age params ##########

### Get medians 
def get_medians(df_p,last):
    df_res = df_p.iloc[-last:].groupby(['param']).median().reset_index()['median'][0]
    return df_res

def medians_params(df_list,age_group,last):    
    params_def = ['age','beta','IFR','RecPeriod','alpha','sigma']
    params_val = [age_group,get_medians(df_list[0],last),get_medians(df_list[1],last),
                  get_medians(df_list[2],last),get_medians(df_list[3],last),get_medians(df_list[4],last)]
    res = dict(zip(params_def,params_val))
    return res

params_data_BOG = pd.read_csv(params_data_path, encoding='unicode_escape', delimiter=',')

# Ages 0-19
young_ages_params = pd.DataFrame(params_data_BOG[params_data_BOG['age_group']=='0-19'])
young_ages_beta = pd.DataFrame(young_ages_params[young_ages_params['param']=='contact_rate'])
young_ages_IFR = pd.DataFrame(young_ages_params[young_ages_params['param']=='IFR'])
young_ages_RecPeriod = pd.DataFrame(young_ages_params[young_ages_params['param']=='recovery_period'])
young_ages_alpha = pd.DataFrame(young_ages_params[young_ages_params['param']=='report_rate'])
young_ages_sigma = pd.DataFrame(young_ages_params[young_ages_params['param']=='relative_asymp_transmission'])
young_params = [young_ages_beta,young_ages_IFR,young_ages_RecPeriod,young_ages_alpha,young_ages_sigma]

# Ages 20-39
youngAdults_ages_params = pd.DataFrame(params_data_BOG[params_data_BOG['age_group']=='20-39'])
youngAdults_ages_beta = pd.DataFrame(youngAdults_ages_params[youngAdults_ages_params['param']=='contact_rate'])
youngAdults_ages_IFR = pd.DataFrame(youngAdults_ages_params[youngAdults_ages_params['param']=='IFR'])
youngAdults_ages_RecPeriod = pd.DataFrame(youngAdults_ages_params[youngAdults_ages_params['param']=='recovery_period'])
youngAdults_ages_alpha = pd.DataFrame(youngAdults_ages_params[youngAdults_ages_params['param']=='report_rate'])
youngAdults_ages_sigma = pd.DataFrame(youngAdults_ages_params[youngAdults_ages_params['param']=='relative_asymp_transmission'])
youngAdults_params = [youngAdults_ages_beta,youngAdults_ages_IFR,youngAdults_ages_RecPeriod,youngAdults_ages_alpha,youngAdults_ages_sigma]

# Ages 40-49
adults_ages_params = pd.DataFrame(params_data_BOG[params_data_BOG['age_group']=='40-49'])
adults_ages_beta = pd.DataFrame(adults_ages_params[adults_ages_params['param']=='contact_rate'])
adults_ages_IFR = pd.DataFrame(adults_ages_params[adults_ages_params['param']=='IFR'])
adults_ages_RecPeriod = pd.DataFrame(adults_ages_params[adults_ages_params['param']=='recovery_period'])
adults_ages_alpha = pd.DataFrame(adults_ages_params[adults_ages_params['param']=='report_rate'])
adults_ages_sigma = pd.DataFrame(adults_ages_params[adults_ages_params['param']=='relative_asymp_transmission'])
adults_params = [adults_ages_beta,adults_ages_IFR,adults_ages_RecPeriod,adults_ages_alpha,adults_ages_sigma]

# Ages 50-59
seniorAdults_ages_params = pd.DataFrame(params_data_BOG[params_data_BOG['age_group']=='50-59'])
seniorAdults_ages_beta = pd.DataFrame(seniorAdults_ages_params[seniorAdults_ages_params['param']=='contact_rate'])
seniorAdults_ages_IFR = pd.DataFrame(seniorAdults_ages_params[seniorAdults_ages_params['param']=='IFR'])
seniorAdults_ages_RecPeriod = pd.DataFrame(seniorAdults_ages_params[seniorAdults_ages_params['param']=='recovery_period'])
seniorAdults_ages_alpha = pd.DataFrame(seniorAdults_ages_params[seniorAdults_ages_params['param']=='report_rate'])
seniorAdults_ages_sigma = pd.DataFrame(seniorAdults_ages_params[seniorAdults_ages_params['param']=='relative_asymp_transmission'])
seniorAdults_params = [seniorAdults_ages_beta,seniorAdults_ages_IFR,seniorAdults_ages_RecPeriod,seniorAdults_ages_alpha,seniorAdults_ages_sigma]

# Ages 60-69
senior_ages_params = pd.DataFrame(params_data_BOG[params_data_BOG['age_group']=='60-69'])
senior_ages_beta = pd.DataFrame(senior_ages_params[senior_ages_params['param']=='contact_rate'])
senior_ages_IFR = pd.DataFrame(senior_ages_params[senior_ages_params['param']=='IFR'])
senior_ages_RecPeriod = pd.DataFrame(senior_ages_params[senior_ages_params['param']=='recovery_period'])
senior_ages_alpha = pd.DataFrame(senior_ages_params[senior_ages_params['param']=='report_rate'])
senior_ages_sigma = pd.DataFrame(senior_ages_params[senior_ages_params['param']=='relative_asymp_transmission'])
senior_params = [senior_ages_beta,senior_ages_IFR,senior_ages_RecPeriod,senior_ages_alpha,senior_ages_sigma]

# Ages 70+
elderly_ages_params = pd.DataFrame(params_data_BOG[params_data_BOG['age_group']=='70-90+'])
elderly_ages_beta = pd.DataFrame(elderly_ages_params[elderly_ages_params['param']=='contact_rate'])
elderly_ages_IFR = pd.DataFrame(elderly_ages_params[elderly_ages_params['param']=='IFR'])
elderly_ages_RecPeriod = pd.DataFrame(elderly_ages_params[elderly_ages_params['param']=='recovery_period'])
elderly_ages_alpha = pd.DataFrame(elderly_ages_params[elderly_ages_params['param']=='report_rate'])
elderly_ages_sigma = pd.DataFrame(elderly_ages_params[elderly_ages_params['param']=='relative_asymp_transmission'])
elderly_params = [elderly_ages_beta,elderly_ages_IFR,elderly_ages_RecPeriod,elderly_ages_alpha,elderly_ages_sigma]


young_params_medians = medians_params(young_params,'0-19',last=15)
youngAdults_params_medians = medians_params(youngAdults_params,'20-39',last=15)
adults_params_medians = medians_params(adults_params,'40-49',last=15)
seniorAdults_params_medians = medians_params(seniorAdults_params,'50-59',last=15)
senior_params_medians = medians_params(senior_params,'60-69',last=15)
elderly_params_medians = medians_params(elderly_params,'70-90+',last=15)





#------------------------------------------------------------------------------------------------------------------------------------

################################
######## Household sizes #######

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

#-----------------------------------------------------------------------------------------------------------------------------------------

###############################
######## Classify nodes #######

preschool_pop_ = preschool_ + teachers_preschool_
preschool_pop = sum(preschool_pop_)

primary_pop_ = primary_ + teachers_primary_
primary_pop = sum(primary_pop_)

highschool_pop_ = highschool_ + teachers_highschool_
highschool_pop = sum(highschool_pop_)

work_pop_no_teachers = sum(work_) - total_teachers_BOG

# Frac of population that is school going, working, preschool or elderly
dist_of_pop = [preschool_pop/total_pop_BOG,
               primary_pop/total_pop_BOG,
               highschool_pop/total_pop_BOG,
               work_pop_no_teachers/total_pop_BOG,
               very_young+university+elderly]

dist_of_pop[-1] += 1-sum(dist_of_pop)

# Classifying each person
classify_pop = np2.random.choice(['preschool','primary','highschool','work','other'], size=pop, p=dist_of_pop)

# Number of individuals in each group
state, counts = np2.unique(classify_pop, return_counts=True)
dict_of_counts = dict(zip(state,counts))
preschool_going = dict_of_counts['preschool']
primary_going = dict_of_counts['primary']
highschool_going = dict_of_counts['highschool']
working = dict_of_counts['work']
other = dict_of_counts['other']

# Indices of individuals in each group
preschool_indx = np2.where(classify_pop=='preschool')[0]
primary_indx = np2.where(classify_pop=='primary')[0]
highschool_indx = np2.where(classify_pop=='highschool')[0]
work_indx = np2.where(classify_pop=='work')[0]
other_indx = np2.where(classify_pop=='other')[0]


# Keep track of the age groups for each individual labelled from 0-16
age_tracker_all = np2.zeros(pop)
age_tracker = np2.zeros(pop)

#------------------------------------------------------------------------------------------------------------------------------------------

###############################
##### Degree distribution #####

### Community --------------------------------------------------------
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


### Preschool -------------------------------------------------------
mean, std = args.preschool_mean, args.preschool_std
p = 1-(std**2/mean)
n_binom = mean/p
preschool_degree = np2.random.binomial(n_binom, p, size = preschool_going)
n_preschool = preschool_going/args.preschool_size
r_preschool = args.preschool_r

preschool_clroom = np2.random.choice(np.arange(0,n_preschool+1,1),size=preschool_going)

# Assign ages to the preschool going population acc. to their proportion from the census data
prob = []
preschool_pop_ = preschool_ + teachers_preschool_
preschool_pop = sum(preschool_pop_)

for i in range(0,len(preschool_pop_)):
    prob.append(preschool_pop_[i]/preschool_pop)
age_group_preschool = np2.random.choice(np.array([1,7]),size=preschool_going,p=prob,replace=True)

for i in range(preschool_going):
    age_tracker[preschool_indx[i]] = age_group_preschool[i]


### Primary ---------------------------------------------------------
mean, std = args.primary_mean, args.primary_std
p = 1-(std**2/mean)
n_binom = mean/p
primary_degree = np2.random.binomial(n_binom, p, size = primary_going)
n_primary = primary_going/args.primary_size
r_primary = args.primary_r

primary_clroom = np2.random.choice(np.arange(0,n_primary+1,1),size=primary_going)

# Assign ages to the primary going population acc. to their proportion from the census data
prob = []
primary_pop_ = primary_ + teachers_primary_
primary_pop = sum(primary_pop_)

for i in range(0,len(primary_pop_)):
    prob.append(primary_pop_[i]/primary_pop)
age_group_primary = np2.random.choice(np.array([2,7]),size=primary_going,p=prob,replace=True)

for i in range(primary_going):
    age_tracker[primary_indx[i]] = age_group_primary[i]


### Highschool -------------------------------------------------------
mean, std = args.highschool_mean, args.highschool_std
p = 1-(std**2/mean)
n_binom = mean/p
highschool_degree = np2.random.binomial(n_binom, p, size = highschool_going)
n_highschool = highschool_going/args.highschool_size
r_highschool = args.highschool_r

highschool_clroom = np2.random.choice(np.arange(0,n_highschool+1,1),size=highschool_going)

# Assign ages to the highschool going population acc. to their proportion from the census data
prob = []
highschool_pop_ = highschool_ + teachers_highschool_
highschool_pop = sum(highschool_pop_)

for i in range(0,len(highschool_pop_)):
    prob.append(highschool_pop_[i]/highschool_pop)
age_group_highschool = np2.random.choice(np.array([3,7]),size=highschool_going,p=prob,replace=True)

for i in range(highschool_going):
    age_tracker[highschool_indx[i]] = age_group_highschool[i]


### Work -----------------------------------------------------------
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

# Split the age group of working population according to the populapreschool_tion seen in the data
p = []
work_pop_ = university_ + work_
work_pop = sum(work_pop_)

for i in range(0,len(work_pop_)):
    p.append(work_pop_[i]/work_pop)
age_group_work = np2.random.choice(np.arange(4,12+1,1),size=working,p=p,replace=True)

for i in range(working):
    age_tracker[work_indx[i]] = age_group_work[i]


#---------------------------------------------------------------------------------------------------------------------------------------

###############################
######## Create graphs ########

print('Creating graphs...')

## Households
matrix_household = create_networks.create_fully_connected(household_sizes,np2.arange(0,pop,1),args.R0,args.MILDINF_DURATION,args.delta_t)

# Get row, col, data information from the sparse matrices
# Converting into DeviceArrays to run faster with jax. Not sure why the lists have to be first converted to usual numpy arrays though
matrix_household_row = np.asarray(np2.asarray(matrix_household[0]))
matrix_household_col = np.asarray(np2.asarray(matrix_household[1]))
matrix_household_data = np.asarray(np2.asarray(matrix_household[2]))

## Preschool
matrix_preschool = create_networks.create_external_corr(pop,preschool_going,preschool_degree,n_preschool,r_preschool,preschool_indx,preschool_clroom,args.R0,args.MILDINF_DURATION,args.delta_t)

matrix_preschool_row = np.asarray(np2.asarray(matrix_preschool[0]))
matrix_preschool_col = np.asarray(np2.asarray(matrix_preschool[2]))
matrix_preschool_data = np.asarray(np2.asarray(matrix_preschool[2]))

## Primary
matrix_primary = create_networks.create_external_corr(pop,primary_going,primary_degree,n_primary,r_primary,primary_indx,primary_clroom,args.R0,args.MILDINF_DURATION,args.delta_t)

matrix_primary_row = np.asarray(np2.asarray(matrix_primary[0]))
matrix_primary_col = np.asarray(np2.asarray(matrix_primary[1]))
matrix_primary_data = np.asarray(np2.asarray(matrix_primary[2]))

## Highschool
matrix_highschool = create_networks.create_external_corr(pop,highschool_going,highschool_degree,n_highschool,r_highschool,highschool_indx,highschool_clroom,args.R0,args.MILDINF_DURATION,args.delta_t)

matrix_highschool_row = np.asarray(np2.asarray(matrix_highschool[0]))
matrix_highschool_col = np.asarray(np2.asarray(matrix_highschool[1]))
matrix_highschool_data = np.asarray(np2.asarray(matrix_highschool[2]))

## Community
matrix_community = create_networks.create_external_corr(pop,pop,community_degree,n_community,r_community,np2.arange(0,pop,1),age_group_community,args.R0,args.MILDINF_DURATION,args.delta_t)

matrix_community_row = np.asarray(np2.asarray(matrix_community[0]))
matrix_community_col = np.asarray(np2.asarray(matrix_community[1]))
matrix_community_data = np.asarray(np2.asarray(matrix_community[2]))

# Saves graphs
multilayer_matrix = [matrix_household,matrix_preschool,matrix_primary,matrix_highschool,matrix_community]


#--------------------------------------------------------------------------------------------------------------------------------------

#########################################
######## Create dynamical layers ########


# Time paramas
Tmax = args.Tmax
days_intervals = [1] * Tmax
delta_t = args.delta_t
step_intervals = [int(x/delta_t) for x in days_intervals]
total_steps = sum(step_intervals)

# Create dynamic
import networks.network_dynamics as nd

print('Creating dynamics...')
if args.school_alternancy:

    time_intervals, ws = nd.create_day_intervention_altern_schools_dynamics(multilayer_matrix,Tmax=Tmax,total_steps=total_steps,schools_day_open=args.school_openings,
                                                            interv_glob=args.intervention,schl_occupation=args.school_occupation,work_occupation=args.work_occupation)

else:

    time_intervals, ws = nd.create_day_intervention_dynamics(multilayer_matrix,Tmax=Tmax,total_steps=total_steps,schools_day_open=args.school_openings,
                                                            interv_glob=args.intervention,schl_occupation=args.school_occupation,work_occupation=args.work_occupation)

#!
# import sys
# sys.path.append('../')

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

config_data = pd.read_csv('config.csv', sep=',', header=None, index_col=0)
figures_path = config_data.loc['figures_dir'][1]
results_path = config_data.loc['results_dir'][1]
params_data_path = config_data.loc['bogota_params_ages_data'][1]
ages_data_path = config_data.loc['bogota_age_data_dir'][1]
houses_data_path = config_data.loc['bogota_houses_data_dir'][1]
teachers_data_path = config_data.loc['bogota_teachers_data_dir'][1]
schools_data_path = config_data.loc['colombia_schools_data_dir'][1]

#from networks import networks
from networks import create_networks

import argparse
parser = argparse.ArgumentParser(description='Simulating interventions')

parser.add_argument('--res_id', default='ND', type=str,
                    help='Result ID for simulation save')

parser.add_argument('--population', default=10000, type=int,
                    help='Speficy the number of individials')
parser.add_argument('--intervention', default=0.6, type=float,
                    help='Intervention efficiancy')
parser.add_argument('--intervention_type', default='intervention', type=str,
                    help='Define the type of intervention [no_intervention,internvention,school_alternancy]')
parser.add_argument('--work_occupation', default=0.6, type=float,
                    help='Percentage of occupation at workplaces over intervention')
parser.add_argument('--school_occupation', default=0.35, type=float,
                    help='Percentage of occupation at classrooms over intervention')
parser.add_argument('--school_openings', default=20, type=int,
                    help='Day of the simulation where schools are open')

parser.add_argument('--ventilation_out', default=3, type=float,
                    help='Ventilation values (h-1) that define how much ventilated is a classroom [2-15]')
parser.add_argument('--fraction_people_masks', default=1.0, type=float,
                    help='Fraction value of people wearing masks')
parser.add_argument('--masks_type', default='N95', type=str,
                    help='Type of masks that individuals are using. Options are: cloth, surgical, N95')
parser.add_argument('--duration_event', default=6, type=float,
                    help='Duration of event (i.e. classes/lectures) in hours over a day')

parser.add_argument('--height_room', default=3.1, type=float,
                    help='Schools height of classroom')
parser.add_argument('--preschool_length_room', default=7.0, type=float,
                    help='Preschool length of classroom')
parser.add_argument('--preschool_width_room', default=7.0, type=float,
                    help='Preschool length of classroom')
parser.add_argument('--primary_length_room', default=10.0, type=float,
                    help='primary length of classroom')
parser.add_argument('--primary_width_room', default=10.0, type=float,
                    help='primary length of classroom')
parser.add_argument('--highschool_length_room', default=10.0, type=float,
                    help='highschool length of classroom')
parser.add_argument('--highschool_width_room', default=10.0, type=float,
                    help='highschool length of classroom')


parser.add_argument('--Tmax', default=200, type=int,
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
recovery_probabilities = np.array([0., 0., FracMild, FracSevere / (FracSevere + FracCritical), 1. - CFR / FracCritical, 0., 0.])

# Define relative infectivity of each state
infection_probabilities = np.array([0., 0., 1.0, 0., 0., 0., 0.])

# Mask efficiencies in inhalation and exhalation taken from https://tinyurl.com/covid-estimator
mask_inhalation = {'cloth':0.5 , 'surgical':0.3, 'N95':0.9}
mask_exhalation = {'cloth':0.5 , 'surgical':0.65, 'N95':0.9}

inhalation_mask = mask_inhalation[args.masks_type]
exhalation_mask = mask_exhalation[args.masks_type]

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
# n_preschool_in_inst = sum(preschool_)/N_schools_BOG  # studens/schools

# Ages 10-14 (2)
primary_ = [int(ages_data_BOG['Total.3'][2].replace('.',''))]
primary = sum(primary_)/total_pop_BOG
# n_primary_in_inst = sum(primary_)/N_schools_BOG  # studens/schools


# Ages 15-19 (3)
highschool_ = [int(ages_data_BOG['Total.3'][3].replace('.',''))]
highschool = sum(highschool_)/total_pop_BOG
# n_highschool_in_inst = sum(highschool_)/N_schools_BOG  # studens/schools

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
young_ages_params    = pd.DataFrame(params_data_BOG[params_data_BOG['age_group']=='0-19'])
young_ages_beta      = pd.DataFrame(young_ages_params[young_ages_params['param']=='contact_rate'])
young_ages_IFR       = pd.DataFrame(young_ages_params[young_ages_params['param']=='IFR'])
young_ages_RecPeriod = pd.DataFrame(young_ages_params[young_ages_params['param']=='recovery_period'])
young_ages_alpha     = pd.DataFrame(young_ages_params[young_ages_params['param']=='report_rate'])
young_ages_sigma     = pd.DataFrame(young_ages_params[young_ages_params['param']=='relative_asymp_transmission'])
young_params = [young_ages_beta,young_ages_IFR,young_ages_RecPeriod,young_ages_alpha,young_ages_sigma]

# Ages 20-39
youngAdults_ages_params    = pd.DataFrame(params_data_BOG[params_data_BOG['age_group']=='20-39'])
youngAdults_ages_beta      = pd.DataFrame(youngAdults_ages_params[youngAdults_ages_params['param']=='contact_rate'])
youngAdults_ages_IFR       = pd.DataFrame(youngAdults_ages_params[youngAdults_ages_params['param']=='IFR'])
youngAdults_ages_RecPeriod = pd.DataFrame(youngAdults_ages_params[youngAdults_ages_params['param']=='recovery_period'])
youngAdults_ages_alpha     = pd.DataFrame(youngAdults_ages_params[youngAdults_ages_params['param']=='report_rate'])
youngAdults_ages_sigma     = pd.DataFrame(youngAdults_ages_params[youngAdults_ages_params['param']=='relative_asymp_transmission'])
youngAdults_params = [youngAdults_ages_beta,youngAdults_ages_IFR,youngAdults_ages_RecPeriod,youngAdults_ages_alpha,youngAdults_ages_sigma]

# Ages 40-49
adults_ages_params    = pd.DataFrame(params_data_BOG[params_data_BOG['age_group']=='40-49'])
adults_ages_beta      = pd.DataFrame(adults_ages_params[adults_ages_params['param']=='contact_rate'])
adults_ages_IFR       = pd.DataFrame(adults_ages_params[adults_ages_params['param']=='IFR'])
adults_ages_RecPeriod = pd.DataFrame(adults_ages_params[adults_ages_params['param']=='recovery_period'])
adults_ages_alpha     = pd.DataFrame(adults_ages_params[adults_ages_params['param']=='report_rate'])
adults_ages_sigma     = pd.DataFrame(adults_ages_params[adults_ages_params['param']=='relative_asymp_transmission'])
adults_params = [adults_ages_beta,adults_ages_IFR,adults_ages_RecPeriod,adults_ages_alpha,adults_ages_sigma]

# Ages 50-59
seniorAdults_ages_params    = pd.DataFrame(params_data_BOG[params_data_BOG['age_group']=='50-59'])
seniorAdults_ages_beta      = pd.DataFrame(seniorAdults_ages_params[seniorAdults_ages_params['param']=='contact_rate'])
seniorAdults_ages_IFR       = pd.DataFrame(seniorAdults_ages_params[seniorAdults_ages_params['param']=='IFR'])
seniorAdults_ages_RecPeriod = pd.DataFrame(seniorAdults_ages_params[seniorAdults_ages_params['param']=='recovery_period'])
seniorAdults_ages_alpha     = pd.DataFrame(seniorAdults_ages_params[seniorAdults_ages_params['param']=='report_rate'])
seniorAdults_ages_sigma     = pd.DataFrame(seniorAdults_ages_params[seniorAdults_ages_params['param']=='relative_asymp_transmission'])
seniorAdults_params = [seniorAdults_ages_beta,seniorAdults_ages_IFR,seniorAdults_ages_RecPeriod,seniorAdults_ages_alpha,seniorAdults_ages_sigma]

# Ages 60-69
senior_ages_params    = pd.DataFrame(params_data_BOG[params_data_BOG['age_group']=='60-69'])
senior_ages_beta      = pd.DataFrame(senior_ages_params[senior_ages_params['param']=='contact_rate'])
senior_ages_IFR       = pd.DataFrame(senior_ages_params[senior_ages_params['param']=='IFR'])
senior_ages_RecPeriod = pd.DataFrame(senior_ages_params[senior_ages_params['param']=='recovery_period'])
senior_ages_alpha     = pd.DataFrame(senior_ages_params[senior_ages_params['param']=='report_rate'])
senior_ages_sigma     = pd.DataFrame(senior_ages_params[senior_ages_params['param']=='relative_asymp_transmission'])
senior_params = [senior_ages_beta,senior_ages_IFR,senior_ages_RecPeriod,senior_ages_alpha,senior_ages_sigma]

# Ages 70+
elderly_ages_params    = pd.DataFrame(params_data_BOG[params_data_BOG['age_group']=='70-90+'])
elderly_ages_beta      = pd.DataFrame(elderly_ages_params[elderly_ages_params['param']=='contact_rate'])
elderly_ages_IFR       = pd.DataFrame(elderly_ages_params[elderly_ages_params['param']=='IFR'])
elderly_ages_RecPeriod = pd.DataFrame(elderly_ages_params[elderly_ages_params['param']=='recovery_period'])
elderly_ages_alpha     = pd.DataFrame(elderly_ages_params[elderly_ages_params['param']=='report_rate'])
elderly_ages_sigma     = pd.DataFrame(elderly_ages_params[elderly_ages_params['param']=='relative_asymp_transmission'])
elderly_params = [elderly_ages_beta,elderly_ages_IFR,elderly_ages_RecPeriod,elderly_ages_alpha,elderly_ages_sigma]


young_params_medians = medians_params(young_params,'0-19',last=15)  # Schools
youngAdults_params_medians = medians_params(youngAdults_params,'20-39',last=15) # Adults
adults_params_medians = medians_params(adults_params,'40-49',last=15)   # Adults
seniorAdults_params_medians = medians_params(seniorAdults_params,'50-59',last=15) # Adults
senior_params_medians = medians_params(senior_params,'60-69',last=15)   # Elders
elderly_params_medians = medians_params(elderly_params,'70-90+',last=15)    # Elders


# Simplify, get medians of values
params_desc = ['age','beta','IFR','RecPeriod','alpha','sigma']

main_adults_params_values  = ['20-59',
                              np2.median([youngAdults_params_medians['beta'],adults_params_medians['beta'],seniorAdults_params_medians['beta']]),
                              np2.median([youngAdults_params_medians['IFR'],adults_params_medians['IFR'],seniorAdults_params_medians['IFR']]),
                              np2.median([youngAdults_params_medians['RecPeriod'],adults_params_medians['RecPeriod'],seniorAdults_params_medians['RecPeriod']]),
                              np2.median([youngAdults_params_medians['alpha'],adults_params_medians['alpha'],seniorAdults_params_medians['alpha']]),
                              np2.median([youngAdults_params_medians['sigma'],adults_params_medians['sigma'],seniorAdults_params_medians['sigma']])]
main_adults_params_medians = dict(zip(params_desc,main_adults_params_values))

main_elders_params_values  = ['60-90+',
                              np2.median([senior_params_medians['beta'],elderly_params_medians['beta']]),
                              np2.median([senior_params_medians['IFR'],elderly_params_medians['IFR']]),
                              np2.median([senior_params_medians['RecPeriod'],elderly_params_medians['RecPeriod']]),
                              np2.median([senior_params_medians['alpha'],elderly_params_medians['alpha']]),
                              np2.median([senior_params_medians['sigma'],elderly_params_medians['sigma']])]
main_elders_params_medians = dict(zip(params_desc,main_elders_params_values))


### Define parameters per layers
def calculate_R0(IFR,alpha,beta,RecPeriod,sigma):
    return (1-IFR)*(alpha*beta*RecPeriod+(1-alpha)*beta*sigma*RecPeriod)

def model_params(params_dict,layer):
    layer_params = {'layer':layer,
                    'RecPeriod':params_dict['RecPeriod'],
                    'R0':calculate_R0(params_dict['IFR'],params_dict['alpha'],params_dict['beta'],
                                      params_dict['RecPeriod'],params_dict['sigma'])}
    return layer_params

school_params = model_params(young_params_medians,'schools')
adults_params = model_params(main_adults_params_medians,'adults')
elders_params = model_params(main_elders_params_medians,'elders')

params_def = ['layer','RecPeriod','R0']
run_params = [ [school_params['layer'],adults_params['layer'],elders_params['layer']],
               [school_params['RecPeriod'],adults_params['RecPeriod'],elders_params['RecPeriod']],
               [school_params['R0'],adults_params['R0'],elders_params['R0']] ]
run_params = dict(zip(params_def,run_params))

df_run_params = pd.DataFrame.from_dict(run_params)

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

#-----------------------------------------------------------------------------------------------------------------------------------

######################################
######## Schools distribution ########

df_schools_COL = pd.read_csv(schools_data_path, 
                    usecols=['DEPARTAMENTO','MUNICIPIO','NOMBRE_ESTABLECIMIENTO','SECTOR','TOTAL_MATRICULA','CANTIDAD_SEDES'])
df_schools_BOG = df_schools_COL[df_schools_COL['DEPARTAMENTO'] == 'CAPITAL BOGOTÁ, D.C.']
N_schools_BOG = len(df_schools_BOG['NOMBRE_ESTABLECIMIENTO'])

# Number of school going in system
school_pop_BOG = sum( preschool_ + primary_ + highschool_ )
school_going_S = preschool_going + primary_going + highschool_going
N_schools_s = int((school_going_S/school_pop_BOG) * N_schools_BOG)

# Separate indices in N schools
preschool_indxs_per_school = np.array_split(preschool_indx, N_schools_s)
primary_indxs_per_school = np.array_split(primary_indx, N_schools_s)
highschool_indxs_per_school = np.array_split(highschool_indx, N_schools_s)

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
r_preschool = args.preschool_r

# Students - Prof proportion
prob = []
preschool_pop_ = preschool_ + teachers_preschool_
preschool_pop = sum(preschool_pop_)

# Assign ages to the preschool going population acc. to their proportion from the census data
for i in range(0,len(preschool_pop_)):
    prob.append(preschool_pop_[i]/preschool_pop)
age_group_preschool = np2.random.choice(np.array([1,7]),size=preschool_going,p=prob,replace=True)

for i in range(preschool_going):
    age_tracker[preschool_indx[i]] = age_group_preschool[i]

preschools_going_sys = []
preschools_nrooms_sys = []
preschools_roomindxs_sys = []
for preschl in range(N_schools_s):
    n_studs = len(preschool_indxs_per_school[preschl])
    n_classrooms = n_studs/args.preschool_size
    preschool_clroom = np2.random.choice(np.arange(0,n_classrooms+1,1),size=n_studs)
    preschools_going_sys.append(n_studs)
    preschools_nrooms_sys.append(int(n_classrooms))
    preschools_roomindxs_sys.append(preschool_clroom)


### Primary ---------------------------------------------------------
mean, std = args.primary_mean, args.primary_std
p = 1-(std**2/mean)
n_binom = mean/p
primary_degree = np2.random.binomial(n_binom, p, size = primary_going)
r_primary = args.primary_r

# Students - Prof proportion
prob = []
primary_pop_ = primary_ + teachers_primary_
primary_pop = sum(primary_pop_)

# Assign ages to the primary going population acc. to their proportion from the census data
for i in range(0,len(primary_pop_)):
    prob.append(primary_pop_[i]/primary_pop)
age_group_primary = np2.random.choice(np.array([1,7]),size=primary_going,p=prob,replace=True)

for i in range(primary_going):
    age_tracker[primary_indx[i]] = age_group_primary[i]

primarys_going_sys = []
primarys_nrooms_sys = []
primarys_roomindxs_sys = []
for preschl in range(N_schools_s):
    n_studs = len(primary_indxs_per_school[preschl])
    n_classrooms = n_studs/args.primary_size
    primary_clroom = np2.random.choice(np.arange(0,n_classrooms+1,1),size=n_studs)
    primarys_going_sys.append(n_studs)
    primarys_nrooms_sys.append(int(n_classrooms))
    primarys_roomindxs_sys.append(primary_clroom)


### Highschool -------------------------------------------------------
mean, std = args.highschool_mean, args.highschool_std
p = 1-(std**2/mean)
n_binom = mean/p
highschool_degree = np2.random.binomial(n_binom, p, size = highschool_going)
r_highschool = args.highschool_r

# Students - Prof proportion
prob = []
highschool_pop_ = highschool_ + teachers_highschool_
highschool_pop = sum(highschool_pop_)

# Assign ages to the highschool going population acc. to their proportion from the census data
for i in range(0,len(highschool_pop_)):
    prob.append(highschool_pop_[i]/highschool_pop)
age_group_highschool = np2.random.choice(np.array([1,7]),size=highschool_going,p=prob,replace=True)

for i in range(highschool_going):
    age_tracker[highschool_indx[i]] = age_group_highschool[i]

highschools_going_sys = []
highschools_nrooms_sys =[]
highschools_roomindxs_sys = []
for preschl in range(N_schools_s):
    n_studs = len(highschool_indxs_per_school[preschl])
    n_classrooms = n_studs/args.highschool_size
    highschool_clroom = np2.random.choice(np.arange(0,n_classrooms+1,1),size=n_studs)
    highschools_going_sys.append(n_studs)
    highschools_nrooms_sys.append(int(n_classrooms))
    highschools_roomindxs_sys.append(highschool_clroom)

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
matrix_household = create_networks.create_fully_connected(household_sizes,age_tracker_all,np2.arange(0,pop,1),df_run_params,args.delta_t)

## Preschool
lst_preschools_matrxs = []
for i in range(N_schools_s):
    matrix_preschool = create_networks.create_external_corr_schools(pop,preschools_going_sys[i],preschool_degree,preschools_nrooms_sys[i],r_preschool,preschool_indxs_per_school[i],preschools_roomindxs_sys[i],age_tracker,df_run_params,args.delta_t
        ,args.preschool_length_room,args.preschool_width_room,args.height_room,args.ventilation_out,inhalation_mask,exhalation_mask,args.fraction_people_masks,args.duration_event)

## Primary
lst_primarys_matrxs = []
for i in range(N_schools_s):
    matrix_primary = create_networks.create_external_corr_schools(pop,primarys_going_sys[i],primary_degree,primarys_nrooms_sys[i],r_primary,primary_indxs_per_school[i],primarys_roomindxs_sys[i],age_tracker,df_run_params,args.delta_t
        ,args.primary_length_room,args.primary_width_room,args.height_room,args.ventilation_out,inhalation_mask,exhalation_mask,args.fraction_people_masks,args.duration_event)

## Highschool
lst_highschools_matrxs = []
for i in range(N_schools_s):
    matrix_highschool = create_networks.create_external_corr_schools(pop,highschools_going_sys[i],highschool_degree,highschools_nrooms_sys[i],r_highschool,highschool_indxs_per_school[i],highschools_roomindxs_sys[i],age_tracker,df_run_params,args.delta_t
        ,args.highschool_length_room,args.highschool_width_room,args.height_room,args.ventilation_out,inhalation_mask,exhalation_mask,args.fraction_people_masks,args.duration_event)

## Work
matrix_work = create_networks.create_external_corr(pop,working,work_degree,n_work,r_work,work_indx,job_place,age_tracker,df_run_params,args.delta_t)

## Community
matrix_community = create_networks.create_external_corr(pop,pop,community_degree,n_community,r_community,np2.arange(0,pop,1),age_group_community,age_tracker,df_run_params,args.delta_t)

# Saves graphs
multilayer_matrix = [matrix_household,matrix_preschool,matrix_primary,matrix_highschool,matrix_work,matrix_community]


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
import networks.network_dynamics_no_interventions as nd_ni


print('Creating dynamics...')

if args.intervention_type == 'no_intervention':
    time_intervals, ws = nd_ni.create_day_intervention_dynamics(multilayer_matrix,Tmax=Tmax,total_steps=total_steps,schools_day_open=0,
                                                            interv_glob=0,schl_occupation=1.0,work_occupation=1.0)

elif args.intervention_type == 'intervention':
    time_intervals, ws = nd.create_day_intervention_dynamics(multilayer_matrix,Tmax=Tmax,total_steps=total_steps,schools_day_open=args.school_openings,
                                                            interv_glob=args.intervention,schl_occupation=args.school_occupation,work_occupation=args.work_occupation)

elif args.intervention_type == 'school_alternancy':
    time_intervals, ws = nd.create_day_intervention_altern_schools_dynamics(multilayer_matrix,Tmax=Tmax,total_steps=total_steps,schools_day_open=args.school_openings,
                                                            interv_glob=args.intervention,schl_occupation=args.school_occupation,work_occupation=args.work_occupation)
else:
    print('No valid intervention type')


#--------------------------------------------------------------------------------------------------------------------------------------

#########################################
############### SIMULATE ################

# Bogota data

cum_cases = 632532
cum_rec   = 593329
mild_house = 17595
hosp_beds = 5369
ICU_beds  = 1351
deaths    = 13125

BOG_E  = int( pop * (cum_cases-cum_rec-mild_house-deaths)/total_pop_BOG)
BOG_R  = int( pop * 0.3 )    # Assuming that 30% of population is already recovered
BOG_I1 = int( pop * mild_house/total_pop_BOG )
BOG_I2 = int( pop * hosp_beds/total_pop_BOG )
BOG_I3 = int( pop * ICU_beds/total_pop_BOG )
BOG_D  = int( pop * deaths/total_pop_BOG )


####################### RUN
print('Simulating...')
soln=np.zeros((args.number_trials,total_steps,7))
soln_cum=np.zeros((args.number_trials,total_steps,7))

for key in tqdm(range(args.number_trials), total=args.number_trials):

  #Initial condition
  init_ind_E = random.uniform(random.PRNGKey(key), shape=(BOG_E,), maxval=pop).astype(np.int32)
  init_ind_I1 = random.uniform(random.PRNGKey(key), shape=(BOG_I1,), maxval=pop).astype(np.int32)
  init_ind_I2 = random.uniform(random.PRNGKey(key), shape=(BOG_I2,), maxval=pop).astype(np.int32)
  init_ind_I3 = random.uniform(random.PRNGKey(key), shape=(BOG_I3,), maxval=pop).astype(np.int32)
  init_ind_D = random.uniform(random.PRNGKey(key), shape=(BOG_D,), maxval=pop).astype(np.int32)
  init_ind_R = random.uniform(random.PRNGKey(key), shape=(BOG_R,), maxval=pop).astype(np.int32)
  init_state = np.zeros(pop, dtype=np.int32)
  init_state = index_update(init_state,init_ind_E,np.ones(BOG_E, dtype=np.int32)*1) # E
  init_state = index_update(init_state,init_ind_I1,np.ones(BOG_I1, dtype=np.int32)*2) # I1
  init_state = index_update(init_state,init_ind_I2,np.ones(BOG_I2, dtype=np.int32)*3) # I2
  init_state = index_update(init_state,init_ind_I3,np.ones(BOG_I3, dtype=np.int32)*4) # I3
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


#------------------------------------------------------------------------------------------------------------------------------------------------

#########################################
############## Save Results #############


# Confidence intervals
loCI = 5
upCI = 95
soln_avg=np.average(soln,axis=0)
soln_loCI=np.percentile(soln,loCI,axis=0)
soln_upCI=np.percentile(soln,upCI,axis=0)

print('Saving results...')

# Save results
tvec = np.linspace(0,Tmax,total_steps)

df_soln_list = []
for i in range(args.number_trials):
  df_results_soln_i = pd.DataFrame(columns=['iter','tvec','S','E','I1','I2','I3','D','R'])
  df_results_soln_i['iter']  = [i] * len(tvec)
  df_results_soln_i['tvec']  = list(tvec)
  df_results_soln_i['S']     = list(soln[i,:,0])
  df_results_soln_i['E']     = list(soln[i,:,1])
  df_results_soln_i['I1']    = list(soln[i,:,2])
  df_results_soln_i['I2']    = list(soln[i,:,3])
  df_results_soln_i['I3']    = list(soln[i,:,4])
  df_results_soln_i['D']     = list(soln[i,:,5])
  df_results_soln_i['R']     = list(soln[i,:,6])
  df_soln_list.append(df_results_soln_i)
df_results_soln = pd.concat(df_soln_list)

df_soln_cum_list = []
for i in range(args.number_trials):
  df_results_soln_cum_i = pd.DataFrame(columns=['iter','tvec','S','E','I1','I2','I3','D','R'])
  df_results_soln_cum_i['iter']  = [i] * len(tvec)
  df_results_soln_cum_i['tvec']  = list(tvec)
  df_results_soln_cum_i['S']     = list(soln_cum[i,:,0])
  df_results_soln_cum_i['E']     = list(soln_cum[i,:,1])
  df_results_soln_cum_i['I1']    = list(soln_cum[i,:,2])
  df_results_soln_cum_i['I2']    = list(soln_cum[i,:,3])
  df_results_soln_cum_i['I3']    = list(soln_cum[i,:,4])
  df_results_soln_cum_i['D']     = list(soln_cum[i,:,5])
  df_results_soln_cum_i['R']     = list(soln_cum[i,:,6])
  df_soln_cum_list.append(df_results_soln_cum_i)
df_results_soln_cum = pd.concat(df_soln_cum_list)


df_results_history = pd.DataFrame(columns=['tvec','S','E','I1','I2','I3','D','R'])
df_results_history['tvec']  = list(tvec)
df_results_history['S']     = list(history[:,0])
df_results_history['E']     = list(history[:,1])
df_results_history['I1']    = list(history[:,2])
df_results_history['I2']    = list(history[:,3])
df_results_history['I3']    = list(history[:,4])
df_results_history['D']     = list(history[:,5])
df_results_history['R']     = list(history[:,6])

df_results_com_history = pd.DataFrame(columns=['tvec','S','E','I1','I2','I3','D','R'])
df_results_com_history['tvec']  = list(tvec)
df_results_com_history['S']     = list(cumulative_history[:,0])
df_results_com_history['E']     = list(cumulative_history[:,1])
df_results_com_history['I1']    = list(cumulative_history[:,2])
df_results_com_history['I2']    = list(cumulative_history[:,3])
df_results_com_history['I3']    = list(cumulative_history[:,4])
df_results_com_history['D']     = list(cumulative_history[:,5])
df_results_com_history['R']     = list(cumulative_history[:,6])


intervention_save = None

if args.intervention_type == 'no_intervention':
    intervention_save = 'no_intervention'

elif args.intervention_type == 'intervention':
    intervention_save = 'intervention'

elif args.intervention_type == 'school_alternancy':
    intervention_save = 'school_alternancy'
       
else:
    print('No valid intervention type')


if not os.path.isdir( os.path.join(results_path, intervention_save, str(number_nodes)) ):
        os.makedirs(os.path.join(results_path, intervention_save, str(number_nodes)))

path_save = os.path.join(results_path, intervention_save, str(number_nodes))

df_results_soln.to_csv(path_save+'/{}_inter_{}_schoolcap_{}_mask_{}_peopleMasked_{}_ventilation_{}_ID_{}_soln.csv'.format(str(number_nodes),str(args.intervention),str(args.school_occupation),args.masks_type,str(args.fraction_people_masks),str(args.ventilation_out),args.res_id), index=False)
df_results_soln_cum.to_csv(path_save+'/{}_inter_{}_schoolcap_{}_mask_{}_peopleMasked_{}_ventilation_{}_ID_{}_soln_cum.csv'.format(str(number_nodes),str(args.intervention),str(args.school_occupation),args.masks_type,str(args.fraction_people_masks),str(args.ventilation_out),args.res_id), index=False)
df_results_history.to_csv(path_save+'/{}_inter_{}_schoolcap_{}_mask_{}_peopleMasked_{}_ventilation_{}_ID_{}_history.csv'.format(str(number_nodes),str(args.intervention),str(args.school_occupation),args.masks_type,str(args.fraction_people_masks),str(args.ventilation_out),args.res_id), index=False)
df_results_com_history.to_csv(path_save+'/{}_inter_{}_schoolcap_{}_mask_{}_peopleMasked_{}_ventilation_{}_ID_{}_com_history.csv'.format(str(number_nodes),str(args.intervention),str(args.school_occupation),args.masks_type,str(args.fraction_people_masks),str(args.ventilation_out),args.res_id), index=False)

print('Done! \n')
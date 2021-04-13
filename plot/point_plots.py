from matplotlib import figure
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from models import model

### Config folders

config_data = pd.read_csv('config.csv', sep=',', header=None, index_col=0)
figures_path = config_data.loc['figures_dir'][1]
results_path = config_data.loc['results_old_dir'][1]
ages_data_path = config_data.loc['bogota_age_data_dir'][1]
houses_data_path = config_data.loc['bogota_houses_data_dir'][1]

### Arguments

import argparse

parser = argparse.ArgumentParser(description='Dynamics visualization.')

parser.add_argument('--population', default=10000, type=int,
                    help='Speficy the number of individials')
parser.add_argument('--type_sim', default='intervention', type=str,
                    help='Speficy the type of simulation to plot')
args = parser.parse_args()

number_nodes = args.population
pop = number_nodes

results_path = os.path.join(results_path,'intervention',str(pop))

### Read functions

def load_results_dyn(type_res,path=results_path,n=pop):
    read_path = os.path.join(path,'{}_{}.csv'.format(str(n),str(type_res)))
    read_file = pd.read_csv(read_path)
    return read_file

def load_results_int(type_res,path=results_path,n=pop):
    read_path = os.path.join(path,'{}_inter_{}_schoolcap_{}_{}.csv'.format(str(n),str(args.intervention),
                                                                           str(args.school_occupation),type_res))
    read_file = pd.read_csv(read_path)
    return read_file

def load_results_ints(type_res,n,int_effec,schl_occup,path=results_path):
    read_path = os.path.join(path,'{}_inter_{}_schoolcap_{}_{}.csv'.format(str(n),str(int_effec),
                                                                           str(schl_occup),type_res))
    read_file = pd.read_csv(read_path)
    return read_file

interv_color_label = ['tab:red','tab:purple','tab:orange']
if not os.path.isdir( os.path.join(figures_path,'point_plots') ):
        os.makedirs( os.path.join(figures_path,'point_plots') )

########################################################################################################################################
########################################################################################################################################
########################################################################################################################################
########################################################################################################################################
########################################################################################################################################

# UCI peaks

intervention_effcs = [0.0,0.2,0.4,0.6]#,1.0]
interv_legend_label = [r'$0\%$ intervention efficiency',r'$20\%$ intervention efficiency',r'$40\%$ intervention efficiency',r'$60\%$ intervention efficiency'] #,r'No intervention, schools $100\%$ occupation']
school_caps        = [0.15,0.25,0.35,0.55,1.0]
states_ = ['S', 'E', 'I1', 'I2', 'I3', 'D', 'R']
df_list = []
for i, inter_ in enumerate(intervention_effcs):
    for j, cap_ in enumerate(school_caps):

        res_read = load_results_ints('soln',args.population,inter_,cap_)
        #res_smooth = model.smooth_timecourse(res_read)

        # if inter_ < 1:
        #     res_read = load_results_ints('soln',args.population,inter_,cap_)
        #     res_smooth = model.smooth_timecourse(res_read)
        # elif inter_ == 1.0:
        #     res_read = load_results_dyn('soln',os.path.join('results','no_intervention',str(pop)))
        #     res_smooth = model.smooth_timecourse(res_read)

        for itr_ in range(10):
            res_smooth_i = res_read['iter'] == itr_
            res_smooth_i = pd.DataFrame(res_read[res_smooth_i])
            peak_I3 = max(res_smooth_i['I3'])*pop

            df_res_i = pd.DataFrame(columns=['iter','interven_eff','school_cap','peak_I3'])
            df_res_i['iter']         = [int(itr_)]
            df_res_i['interven_eff'] = r'{}$\%$'.format(int(inter_*100))
            df_res_i['school_cap']   = int(cap_*100)
            df_res_i['peak_I3']      = peak_I3
            df_list.append(df_res_i)

df_peaks_I3 = pd.concat(df_list)


fig,ax = plt.subplots(1,1,figsize=(7, 6))
sns.pointplot(ax=ax, data=df_peaks_I3, x='school_cap', y='peak_I3', hue='interven_eff', linestyles='--',palette='viridis',alpha=0.5)
#plt.legend(interv_legend_label,frameon=False,framealpha=0.0,bbox_to_anchor=(0,1), loc="lower center")
ax.legend().set_title('')
plt.setp(ax.get_legend().get_texts(), fontsize='17') # for legend text
ax.set_xlabel(r'School capacity ($\%$)',fontsize=17)
ax.set_ylabel(r'Beds per 10,000',fontsize=17)
ax.set_title(r'ICUs required in peak',fontsize=17)
plt.xticks(size=17)
plt.yticks(size=17)
save_path = os.path.join(figures_path,'point_plots','ICU_peakbeds_n_{}.png'.format(str(pop)))
plt.savefig(save_path,dpi=400, transparent=True, bbox_inches='tight', pad_inches=0.1 )

########################################################################################################################################
########################################################################################################################################
########################################################################################################################################
########################################################################################################################################
########################################################################################################################################

# No-UCI peaks
intervention_effcs = [0.0,0.2,0.4,0.6]#,1.0]
interv_legend_label = [r'$0\%$ intervention efficiency',r'$20\%$ intervention efficiency',r'$40\%$ intervention efficiency',r'$60\%$ intervention efficiency'] #,r'No intervention, schools $100\%$ occupation']
school_caps        = [0.15,0.25,0.35,0.55,1.0]
states_ = ['S', 'E', 'I1', 'I2', 'I3', 'D', 'R']
df_list = []
for i, inter_ in enumerate(intervention_effcs):
    for j, cap_ in enumerate(school_caps):

        res_read = load_results_ints('soln',args.population,inter_,cap_)
        #res_smooth = model.smooth_timecourse(res_read)

        # if inter_ < 1:
        #     res_read = load_results_ints('soln',args.population,inter_,cap_)
        #     res_smooth = model.smooth_timecourse(res_read)
        # elif inter_ == 1.0:
        #     res_read = load_results_dyn('soln',os.path.join('results','no_intervention',str(pop)))
        #     res_smooth = model.smooth_timecourse(res_read)

        for itr_ in range(10):
            res_smooth_i = res_read['iter'] == itr_
            res_smooth_i = pd.DataFrame(res_read[res_smooth_i])
            peak_I2 = max(res_smooth_i['I2'])*pop

            df_res_i = pd.DataFrame(columns=['iter','interven_eff','school_cap','peak_I2'])
            df_res_i['iter']         = [int(itr_)]
            df_res_i['interven_eff'] = r'{}$\%$'.format(int(inter_*100))
            df_res_i['school_cap']   = int(cap_*100)
            df_res_i['peak_I2']      = peak_I2
            df_list.append(df_res_i)

df_peaks_I2 = pd.concat(df_list)


fig,ax = plt.subplots(1,1,figsize=(7, 6))
sns.pointplot(ax=ax, data=df_peaks_I2, x='school_cap', y='peak_I2', hue='interven_eff', linestyles='--',palette='viridis',alpha=0.5)
#plt.legend(interv_legend_label,frameon=False,framealpha=0.0,bbox_to_anchor=(0,1), loc="lower center")
ax.legend().set_title('')
plt.setp(ax.get_legend().get_texts(), fontsize='17') # for legend text
ax.set_xlabel(r'School capacity ($\%$)',fontsize=17)
ax.set_ylabel(r'Beds per 10,000',fontsize=17)
ax.set_title(r'Non-ICUs beds required in peak',fontsize=17)
plt.xticks(size=17)
plt.yticks(size=17)
save_path = os.path.join(figures_path,'point_plots','nonICU_peakbeds_n_{}.png'.format(str(pop)))
plt.savefig(save_path,dpi=400, transparent=True, bbox_inches='tight', pad_inches=0.1 )



########################################################################################################################################
########################################################################################################################################
########################################################################################################################################
########################################################################################################################################
########################################################################################################################################

# Final deaths

intervention_effcs = [0.0,0.2,0.4,0.6]
interv_legend_label = [r'$0\%$ intervention efficiency',r'$20\%$ intervention efficiency',r'$40\%$ intervention efficiency',r'$60\%$ intervention efficiency'] #,r'No intervention, schools $100\%$ occupation']
school_caps        = [0.0,0.15,0.25,0.35,0.55]
states_ = ['S', 'E', 'I1', 'I2', 'I3', 'D', 'R']
df_list = []
for i, inter_ in enumerate(intervention_effcs):
    for j, cap_ in enumerate(school_caps):

        res_read = load_results_ints('soln_cum',args.population,inter_,cap_)

        for itr_ in range(10):
            res_read_i = res_read['iter'] == itr_
            res_read_i = pd.DataFrame(res_read[res_read_i])
            end_dead = res_read_i['D'].iloc[-1]

            df_res_i = pd.DataFrame(columns=['iter','interven_eff','school_cap','end_dead'])
            df_res_i['iter']         = [int(itr_)]
            df_res_i['interven_eff'] = r'{}$\%$'.format(int(inter_*100))
            df_res_i['school_cap']   = int(cap_*100)
            df_res_i['end_dead']      = end_dead*pop
            df_list.append(df_res_i)

df_peaks_D = pd.concat(df_list)


fig,ax = plt.subplots(1,1,figsize=(7, 6))
sns.pointplot(ax=ax, data=df_peaks_D, x='school_cap', y='end_dead', hue='interven_eff', linestyles='',palette='viridis',alpha=0.5)
#plt.legend(interv_legend_label,frameon=False,framealpha=0.0,bbox_to_anchor=(0,1), loc="lower center")
ax.legend(bbox_to_anchor=(1.02,1)).set_title('Intervention efficiency')
plt.setp(ax.get_legend().get_texts(), fontsize='17') # for legend text
ax.set_xlabel(r'School capacity ($\%$)',fontsize=17)
ax.set_ylabel(r'Deaths per 10,000',fontsize=17)
ax.set_title(r'Total deaths',fontsize=17)
plt.xticks(size=17)
plt.yticks(size=17)
#plt.show()
save_path = os.path.join(figures_path,'point_plots','totalDeaths_n_{}_55.png'.format(str(pop)))
plt.savefig(save_path,dpi=400, transparent=True, bbox_inches='tight', pad_inches=0.1 )



########################################################################################################################################
########################################################################################################################################
########################################################################################################################################
########################################################################################################################################
########################################################################################################################################

# Final cases

intervention_effcs = [0.0,0.2,0.4,0.6]
interv_legend_label = [r'$0\%$ intervention efficiency',r'$20\%$ intervention efficiency',r'$40\%$ intervention efficiency',r'$60\%$ intervention efficiency'] #,r'No intervention, schools $100\%$ occupation']
school_caps        = [0.0,0.15,0.25,0.35,0.55]
states_ = ['S', 'E', 'I1', 'I2', 'I3', 'D', 'R']
df_list = []
for i, inter_ in enumerate(intervention_effcs):
    for j, cap_ in enumerate(school_caps):

        res_read = load_results_ints('soln_cum',args.population,inter_,cap_)

        for itr_ in range(10):
            res_read_i = res_read['iter'] == itr_
            res_read_i = pd.DataFrame(res_read[res_read_i])
            end_cases = res_read_i['E'].iloc[-1]

            df_res_i = pd.DataFrame(columns=['iter','interven_eff','school_cap','end_cases'])
            df_res_i['iter']         = [int(itr_)]
            df_res_i['interven_eff'] = r'{}$\%$'.format(int(inter_*100))
            df_res_i['school_cap']   = int(cap_*100)
            df_res_i['end_cases']      = end_cases*pop
            df_list.append(df_res_i)

df_peaks_E = pd.concat(df_list)


fig,ax = plt.subplots(1,1,figsize=(7, 6))
sns.pointplot(ax=ax, data=df_peaks_E, x='school_cap', y='end_cases', hue='interven_eff', linestyles='',palette='viridis',alpha=0.5)
#plt.legend(interv_legend_label,frameon=False,framealpha=0.0,bbox_to_anchor=(0,1), loc="lower center")
ax.legend(bbox_to_anchor=(1.02,1)).set_title('')
plt.setp(ax.get_legend().get_texts(), fontsize='17') # for legend text
ax.set_xlabel(r'School capacity ($\%$)',fontsize=17)
ax.set_ylabel(r'Infections per 10,000',fontsize=17)
ax.set_title(r'Total infections',fontsize=17)
plt.xticks(size=17)
plt.yticks(size=17)
#plt.show()
save_path = os.path.join(figures_path,'point_plots','totalInfections_n_{}_55.png'.format(str(pop)))
plt.savefig(save_path,dpi=400, transparent=True, bbox_inches='tight', pad_inches=0.1 )
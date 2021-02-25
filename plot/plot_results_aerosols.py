import sys
sys.path.append('../')

from matplotlib import figure
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from models import model

### Config folders

config_data = pd.read_csv('configlin.csv', sep=',', header=None, index_col=0)
figures_path = config_data.loc['figures_dir'][1]
results_path = config_data.loc['results_dir'][1]
ages_data_path = config_data.loc['bogota_age_data_dir'][1]
houses_data_path = config_data.loc['bogota_houses_data_dir'][1]

### Arguments

import argparse

parser = argparse.ArgumentParser(description='Dynamics visualization.')

parser.add_argument('--population', default=5000, type=int,
                    help='Speficy the number of individials')
parser.add_argument('--type_sim', default='intervention', type=str,
                    help='Speficy the type of simulation to plot')
args = parser.parse_args()

number_nodes = args.population
pop = number_nodes

### Read functions

def load_results_ints(type_res,n,int_effec,schl_occup,type_mask,frac_people_mask,ventilation,path=results_path):
    read_path = os.path.join(path,'{}_inter_{}_schoolcap_{}_mask_{}_peopleMasked_{}_ventilation_{}_ID_ND_{}.csv'.format(str(n),str(int_effec),
                                                                           str(schl_occup),type_mask,str(frac_people_mask),str(ventilation),type_res))
    read_file = pd.read_csv(read_path)
    return read_file


### Read file

results_path = os.path.join(results_path,'intervention',str(pop))


###------------------------------------------------------------------------------------------------------------------------------------------------------

### Plot new cases

# School capacity of 35%, low ventilation, different masks

intervention_effcs = [0.0,0.2] #,0.4,0.6]
interv_legend_label = [r'$0\%$ intervention efficiency',r'$20\%$ intervention efficiency'] #,r'$40\%$ intervention efficiency',r'$60\%$ intervention efficiency'] #,r'No intervention, schools $100\%$ occupation']
interv_color_label = ['k','tab:red'] #,'tab:purple','tab:orange']

school_caps = [0.35]

masks = ['cloth','surgical','N95']

ventilation_val = 0.0

fraction_people_masked = 1.0

plot_state = 'E'

alpha = 0.05
for mask_ in masks:
    for c, cap_ in tqdm(enumerate(school_caps), total=len(school_caps)):
        plt.figure(figsize=(6,4))  # create figure
        for i, inter_ in enumerate(intervention_effcs):
            # read results
            res_read = load_results_ints('soln',args.population,inter_,cap_,mask_,fraction_people_masked,ventilation_val,path=results_path)
            res_median = res_read.groupby('tvec').median(); res_median = res_median.reset_index()
            res_loCI = res_read.groupby('tvec').quantile(alpha/2); res_loCI = res_loCI.reset_index()
            res_upCI = res_read.groupby('tvec').quantile(1-alpha/2); res_upCI = res_upCI.reset_index()

            plt.plot(res_median['tvec'],res_median[plot_state]*pop,color=interv_color_label[i])
            plt.legend(interv_legend_label,frameon=False,framealpha=0.0,bbox_to_anchor=(1,1), loc="best")
            plt.gca().set_prop_cycle(None)
            plt.fill_between(res_median['tvec'],res_loCI[plot_state]*pop,res_upCI[plot_state]*pop,color=interv_color_label[i],alpha=0.3)
            plt.axvspan(0,20,color='gray',alpha=0.05)
            plt.annotate('Schools \n closed',(0,500),size=9)
            plt.annotate('Schools \n open',(22,500),size=9)
            plt.xlim([0,max(res_median['tvec'])])
            plt.ylim([0,0.065*pop])
            plt.xticks(size=12)
            plt.yticks(size=12)
            plt.xlabel("Time (days)",size=12)
            plt.ylabel(r"New cases per 10,000 ind",size=12)
            if args.type_sim == 'intervention':
                plt.title(r'Schools at ${}\%$, low ventilation, all wearing {} masks'.format(int(cap_*100),mask_))
            elif args.type_sim == 'school_alternancy':
                plt.title(r'New cases with schools alterning ${}\%$ occupation'.format(int(cap_*100)))

    if not os.path.isdir( os.path.join(figures_path,'cases_evolution') ):
        os.makedirs( os.path.join(figures_path,'cases_evolution') )

    save_path = os.path.join(figures_path,'cases_evolution','{}_lin_{}_dynamics_schoolcap_{}_n_{}_mask_{}_peopleMasked_{}_ventilation_{}.png'.format(
                    plot_state,args.type_sim,cap_,str(pop),mask_,str(fraction_people_masked),str(ventilation_val)))
    plt.savefig(save_path,dpi=400, transparent=True, bbox_inches='tight', pad_inches=0.1 )
    #plt.show()


plot_state = 'D'

alpha = 0.05
for mask_ in masks:
    for c, cap_ in tqdm(enumerate(school_caps), total=len(school_caps)):
        plt.figure(figsize=(6,4))  # create figure
        for i, inter_ in enumerate(intervention_effcs):
            # read results
            res_read = load_results_ints('soln',args.population,inter_,cap_,mask_,fraction_people_masked,ventilation_val,path=results_path)
            res_median = res_read.groupby('tvec').median(); res_median = res_median.reset_index()
            res_loCI = res_read.groupby('tvec').quantile(alpha/2); res_loCI = res_loCI.reset_index()
            res_upCI = res_read.groupby('tvec').quantile(1-alpha/2); res_upCI = res_upCI.reset_index()

            plt.plot(res_median['tvec'],res_median[plot_state]*pop,color=interv_color_label[i])
            plt.legend(interv_legend_label,frameon=False,framealpha=0.0,bbox_to_anchor=(1,1), loc="best")
            plt.gca().set_prop_cycle(None)
            plt.fill_between(res_median['tvec'],res_loCI[plot_state]*pop,res_upCI[plot_state]*pop,color=interv_color_label[i],alpha=0.3)
            plt.axvspan(0,20,color='gray',alpha=0.05)
            plt.annotate('Schools \n closed',(0,500),size=9)
            plt.annotate('Schools \n open',(22,500),size=9)
            plt.xlim([0,max(res_median['tvec'])])
            plt.ylim([0,0.02*pop])
            plt.xticks(size=12)
            plt.yticks(size=12)
            plt.xlabel("Time (days)",size=12)
            plt.ylabel(r"Deaths per 10,000 ind",size=12)
            if args.type_sim == 'intervention':
                plt.title(r'Schools at ${}\%$, low ventilation, all wearing {} masks'.format(int(cap_*100),mask_))
            elif args.type_sim == 'school_alternancy':
                plt.title(r'New cases with schools alterning ${}\%$ occupation'.format(int(cap_*100)))

    if not os.path.isdir( os.path.join(figures_path,'cases_evolution') ):
        os.makedirs( os.path.join(figures_path,'cases_evolution') )

    save_path = os.path.join(figures_path,'cases_evolution','{}_lin_{}_dynamics_schoolcap_{}_n_{}_mask_{}_peopleMasked_{}_ventilation_{}.png'.format(
                    plot_state,args.type_sim,cap_,str(pop),mask_,str(fraction_people_masked),str(ventilation_val)))
    plt.savefig(save_path,dpi=400, transparent=True, bbox_inches='tight', pad_inches=0.1 )
    #plt.show()


# School capacity of 35%, high ventilation, different masks

intervention_effcs = [0.0,0.2] #,0.4,0.6]
interv_legend_label = [r'$0\%$ intervention efficiency',r'$20\%$ intervention efficiency'] #,r'$40\%$ intervention efficiency',r'$60\%$ intervention efficiency'] #,r'No intervention, schools $100\%$ occupation']
interv_color_label = ['k','tab:red'] #,'tab:purple','tab:orange']

school_caps = [0.35]

masks = ['cloth','surgical','N95']

ventilation_val = 15.0

fraction_people_masked = 1.0

plot_state = 'E'

alpha = 0.05
for mask_ in masks:
    for c, cap_ in tqdm(enumerate(school_caps), total=len(school_caps)):
        plt.figure(figsize=(6,4))  # create figure
        for i, inter_ in enumerate(intervention_effcs):
            # read results
            res_read = load_results_ints('soln',args.population,inter_,cap_,mask_,fraction_people_masked,ventilation_val,path=results_path)
            res_median = res_read.groupby('tvec').median(); res_median = res_median.reset_index()
            res_loCI = res_read.groupby('tvec').quantile(alpha/2); res_loCI = res_loCI.reset_index()
            res_upCI = res_read.groupby('tvec').quantile(1-alpha/2); res_upCI = res_upCI.reset_index()

            plt.plot(res_median['tvec'],res_median[plot_state]*pop,color=interv_color_label[i])
            plt.legend(interv_legend_label,frameon=False,framealpha=0.0,bbox_to_anchor=(1,1), loc="best")
            plt.gca().set_prop_cycle(None)
            plt.fill_between(res_median['tvec'],res_loCI[plot_state]*pop,res_upCI[plot_state]*pop,color=interv_color_label[i],alpha=0.3)
            plt.axvspan(0,20,color='gray',alpha=0.05)
            plt.annotate('Schools \n closed',(0,500),size=9)
            plt.annotate('Schools \n open',(22,500),size=9)
            plt.xlim([0,max(res_median['tvec'])])
            plt.ylim([0,0.065*pop])
            plt.xticks(size=12)
            plt.yticks(size=12)
            plt.xlabel("Time (days)",size=12)
            plt.ylabel(r"New cases per 10,000 ind",size=12)
            if args.type_sim == 'intervention':
                plt.title(r'Schools at ${}\%$, high ventilation, all wearing {} masks'.format(int(cap_*100),mask_))
            elif args.type_sim == 'school_alternancy':
                plt.title(r'New cases with schools alterning ${}\%$ occupation'.format(int(cap_*100)))

    if not os.path.isdir( os.path.join(figures_path,'cases_evolution') ):
        os.makedirs( os.path.join(figures_path,'cases_evolution') )

    save_path = os.path.join(figures_path,'cases_evolution','{}_lin_{}_dynamics_schoolcap_{}_n_{}_mask_{}_peopleMasked_{}_ventilation_{}.png'.format(
                    plot_state,args.type_sim,cap_,str(pop),mask_,str(fraction_people_masked),str(ventilation_val)))
    plt.savefig(save_path,dpi=400, transparent=True, bbox_inches='tight', pad_inches=0.1 )
    #plt.show()


plot_state = 'D'

alpha = 0.05
for mask_ in masks:
    for c, cap_ in tqdm(enumerate(school_caps), total=len(school_caps)):
        plt.figure(figsize=(6,4))  # create figure
        for i, inter_ in enumerate(intervention_effcs):
            # read results
            res_read = load_results_ints('soln',args.population,inter_,cap_,mask_,fraction_people_masked,ventilation_val,path=results_path)
            res_median = res_read.groupby('tvec').median(); res_median = res_median.reset_index()
            res_loCI = res_read.groupby('tvec').quantile(alpha/2); res_loCI = res_loCI.reset_index()
            res_upCI = res_read.groupby('tvec').quantile(1-alpha/2); res_upCI = res_upCI.reset_index()

            plt.plot(res_median['tvec'],res_median[plot_state]*pop,color=interv_color_label[i])
            plt.legend(interv_legend_label,frameon=False,framealpha=0.0,bbox_to_anchor=(1,1), loc="best")
            plt.gca().set_prop_cycle(None)
            plt.fill_between(res_median['tvec'],res_loCI[plot_state]*pop,res_upCI[plot_state]*pop,color=interv_color_label[i],alpha=0.3)
            plt.axvspan(0,20,color='gray',alpha=0.05)
            plt.annotate('Schools \n closed',(0,500),size=9)
            plt.annotate('Schools \n open',(22,500),size=9)
            plt.xlim([0,max(res_median['tvec'])])
            plt.ylim([0,0.02*pop])
            plt.xticks(size=12)
            plt.yticks(size=12)
            plt.xlabel("Time (days)",size=12)
            plt.ylabel(r"Deaths per 10,000 ind",size=12)
            if args.type_sim == 'intervention':
                plt.title(r'Schools at ${}\%$, high ventilation, all wearing {} masks'.format(int(cap_*100),mask_))
            elif args.type_sim == 'school_alternancy':
                plt.title(r'New cases with schools alterning ${}\%$ occupation'.format(int(cap_*100)))

    if not os.path.isdir( os.path.join(figures_path,'cases_evolution') ):
        os.makedirs( os.path.join(figures_path,'cases_evolution') )

    save_path = os.path.join(figures_path,'cases_evolution','{}_lin_{}_dynamics_schoolcap_{}_n_{}_mask_{}_peopleMasked_{}_ventilation_{}.png'.format(
                    plot_state,args.type_sim,cap_,str(pop),mask_,str(fraction_people_masked),str(ventilation_val)))
    plt.savefig(save_path,dpi=400, transparent=True, bbox_inches='tight', pad_inches=0.1 )
    #plt.show()


###------------------------------------------------------------------------------------------------------------------------------------------------------

### Plot point plots

# Cases cumulative

intervention_effcs = [0.0,0.4]
interv_legend_label = [r'$0\%$ intervention efficiency',r'$40\%$ intervention efficiency'] #,r'No intervention, schools $100\%$ occupation']

ventilation_vals = [2,11]

states_ = ['S', 'E', 'I1', 'I2', 'I3', 'D', 'R']
df_list = []

for i, inter_ in enumerate(intervention_effcs):
    for j, vent_ in enumerate(ventilation_vals):

        res_read = load_results_ints('soln_cum',args.population,inter_,0.35,mask_,fraction_people_masked,ventilation_val,path=results_path)

        for itr_ in range(10):
            res_read_i = res_read['iter'] == itr_
            res_read_i = pd.DataFrame(res_read[res_read_i])
            end_cases = res_read_i['E'].iloc[-1]

            df_res_i = pd.DataFrame(columns=['iter','interven_eff','ventilation','end_cases'])
            df_res_i['iter']         = [int(itr_)]
            df_res_i['interven_eff'] = r'{}$\%$'.format(int(inter_*100))
            df_res_i['ventilation']   = int(vent_)
            df_res_i['end_cases']      = end_cases*pop
            df_list.append(df_res_i)

df_peaks_E = pd.concat(df_list)


fig,ax = plt.subplots(1,1,figsize=(7, 6))
sns.pointplot(ax=ax, data=df_peaks_E, x='ventilation', y='end_cases', hue='interven_eff', linestyles='',palette='viridis',alpha=0.5)
#plt.legend(interv_legend_label,frameon=False,framealpha=0.0,bbox_to_anchor=(0,1), loc="lower center")
ax.legend(bbox_to_anchor=(1.02,1)).set_title('Intervention efficiency')
plt.setp(ax.get_legend().get_texts(), fontsize='17') # for legend text
ax.set_xlabel(r'Ventilation rate (h-1)',fontsize=17)
ax.set_ylabel(r'Infections per 10,000',fontsize=17)
ax.set_title(r'Total infections',fontsize=17)
plt.xticks(size=17)
plt.yticks(size=17)
#plt.show()
save_path = os.path.join(figures_path,'point_plots','totalInfections_n_{}_schoolcap_{}_.png'.format(str(pop),str(0.35)))
plt.savefig(save_path,dpi=400, transparent=True, bbox_inches='tight', pad_inches=0.1 )



###------------------------------------------------------------------------------------------------------------------------------------------------------
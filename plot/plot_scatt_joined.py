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

config_data = pd.read_csv('config.csv', sep=',', header=None, index_col=0)
figures_path = config_data.loc['figures_dir'][1]
results_path = config_data.loc['results_dir'][1]
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

### Read functions

def load_results_ints(type_res,n,int_effec,schl_occup,type_mask,frac_people_mask,ventilation,path=results_path):
    read_path = os.path.join(path,'{}_inter_{}_schoolcap_{}_mask_{}_peopleMasked_{}_ventilation_{}_ID_ND_{}.csv'.format(str(n),str(int_effec),
                                                                           str(schl_occup),type_mask,str(frac_people_mask),str(ventilation),type_res))
    read_file = pd.read_csv(read_path)
    return read_file


### Read file

results_path = os.path.join(results_path,'intervention',str(pop))



### Point plots

intervention_effcs = [0.0,0.2,0.4]
inter_ = intervention_effcs[2]
interv_legend_label = [r'$0\%$ intervention efficiency',r'$20\%$ intervention efficiency',r'$40\%$ intervention efficiency'] #,r'$40\%$ intervention efficiency',r'$60\%$ intervention efficiency'] #,r'No intervention, schools $100\%$ occupation']

school_cap = 0.35

fraction_people_masked = [0.5,0.65,0.8,0.95,1.0]

ventilation_vals = [0.0, 15.0]

masks = ['cloth','surgical','N95']
masks_labels = ['Cloth','Surgical','N95']
masks_labels = dict(zip(masks,masks_labels))

df_list = []
fig,ax = plt.subplots(1,1,figsize=(7, 6))
for i, vent_ in enumerate(ventilation_vals):
    for m, mask_ in enumerate(masks):
        for j, frac_mask_ in enumerate(fraction_people_masked):

            res_read = load_results_ints('soln_cum',args.population,inter_,school_cap,mask_,frac_mask_,vent_,path=results_path)
            for itr_ in range(10):
                res_read_i = res_read['iter'] == itr_
                res_read_i = pd.DataFrame(res_read[res_read_i])
                end_cases = res_read_i['E'].iloc[-1]

                df_res_i = pd.DataFrame(columns=['iter','mask','frac_mask','interven_eff','ventilation','end_cases'])
                df_res_i['iter']         = [int(itr_)]
                df_res_i['mask']         = masks_labels[mask_]
                df_res_i['frac_mask']    = r'{}%'.format(int(frac_mask_*100))
                df_res_i['interven_eff'] = r'{}%'.format(int(inter_*100))
                df_res_i['ventilation']   = str(vent_)
                df_res_i['end_cases']      = end_cases*100
                df_list.append(df_res_i)

    df_final_E_v = pd.concat(df_list)
    sns.pointplot(ax=ax, data=df_final_E_v, x='end_cases', y='frac_mask', hue='mask', linestyles='', markers='^', dodge=0.3,palette='plasma',alpha=0.8)
    ax.legend(bbox_to_anchor=(1.02,1)).set_title('')
    ax.get_legend().remove()
    #plt.setp(ax.get_legend().get_texts(), fontsize='17') # for legend text
    ax.set_xlabel(r'Infections ($\%$)',fontsize=17)
    ax.set_ylabel(r'Individuals wearing masks ($\%$)',fontsize=17)
    ax.set_title(r'Total infections | schools at {}$\%$'.format(str(school_cap*100)),fontsize=17)
    plt.xticks(size=16)
    plt.yticks(size=16)
    #plt.xlim([350,4100])
plt.show()

#---------------------------------------------------------
#---------------------------------------------------------

intervention_effcs = [0.0,0.2,0.4]
inter_ = intervention_effcs[2]
interv_legend_label = [r'$0\%$ intervention efficiency',r'$20\%$ intervention efficiency',r'$40\%$ intervention efficiency'] #,r'$40\%$ intervention efficiency',r'$60\%$ intervention efficiency'] #,r'No intervention, schools $100\%$ occupation']

school_cap = 0.35

fraction_people_masked = [0.5,0.65,0.8,0.95,1.0]

ventilation_vals = [0.0, 15.0]

masks = ['cloth','surgical','N95']
masks_labels = ['Cloth','Surgical','N95']
masks_labels = dict(zip(masks,masks_labels))

df_list = []
for m, mask_ in enumerate(masks):
    for j, frac_mask_ in enumerate(fraction_people_masked):

        res_read = load_results_ints('soln_cum',args.population,inter_,school_cap,mask_,frac_mask_,ventilation_vals[0],path=results_path)
        for itr_ in range(10):
            res_read_i = res_read['iter'] == itr_
            res_read_i = pd.DataFrame(res_read[res_read_i])
            end_cases = res_read_i['E'].iloc[-1]

            df_res_i = pd.DataFrame(columns=['iter','mask','frac_mask','interven_eff','ventilation','end_cases'])
            df_res_i['iter']         = [int(itr_)]
            df_res_i['mask']         = masks_labels[mask_]
            df_res_i['frac_mask']    = r'{}%'.format(int(frac_mask_*100))
            df_res_i['interven_eff'] = r'{}%'.format(int(inter_*100))
            df_res_i['ventilation']   = str(vent_)
            df_res_i['end_cases']      = end_cases*100
            df_list.append(df_res_i)

df_final_E_Lv = pd.concat(df_list)

df_list = []
for m, mask_ in enumerate(masks):
    for j, frac_mask_ in enumerate(fraction_people_masked):

        res_read = load_results_ints('soln_cum',args.population,inter_,school_cap,mask_,frac_mask_,ventilation_vals[1],path=results_path)
        for itr_ in range(10):
            res_read_i = res_read['iter'] == itr_
            res_read_i = pd.DataFrame(res_read[res_read_i])
            end_cases = res_read_i['E'].iloc[-1]

            df_res_i = pd.DataFrame(columns=['iter','mask','frac_mask','interven_eff','ventilation','end_cases'])
            df_res_i['iter']         = [int(itr_)]
            df_res_i['mask']         = masks_labels[mask_]
            df_res_i['frac_mask']    = r'{}%'.format(int(frac_mask_*100))
            df_res_i['interven_eff'] = r'{}%'.format(int(inter_*100))
            df_res_i['ventilation']   = str(vent_)
            df_res_i['end_cases']      = end_cases*100
            df_list.append(df_res_i)

df_final_E_Hv = pd.concat(df_list)


fig,ax = plt.subplots(1,1,figsize=(7, 6))
sns.pointplot(ax=ax, data=df_final_E_Lv, x='end_cases', y='frac_mask', hue='mask', linestyles='', markers='o', dodge=0.3,palette='plasma',alpha=0.8)
ax.legend(bbox_to_anchor=(1.02,1)).set_title('')
ax.get_legend().remove()
sns.pointplot(ax=ax, data=df_final_E_Hv, x='end_cases', y='frac_mask', hue='mask', linestyles='', markers='^', dodge=0.3,palette='plasma',alpha=0.8)
ax.legend(bbox_to_anchor=(1.02,1)).set_title('')
ax.get_legend().remove()
#plt.setp(ax.get_legend().get_texts(), fontsize='17') # for legend text
ax.set_xlabel(r'Infections ($\%$)',fontsize=17)
ax.set_ylabel(r'Individuals wearing masks ($\%$)',fontsize=17)
ax.set_title(r'Total infections | schools at {}$\%$'.format(str(school_cap*100)),fontsize=17)
plt.xticks(size=16)
plt.yticks(size=16)
plt.xlim([0,41])
save_path = os.path.join(figures_path,'point_plots','totalInfectionsPCT_n_{}_schoolcap_{}_ventilation_Lo&Hi_inter_{}.png'.format(str(pop),str(0.35),str(ventilation_vals[0]),str(inter_)))
plt.savefig(save_path,dpi=400, transparent=False, bbox_inches='tight', pad_inches=0.1 )

#---------------------------------------------------------
#---------------------------------------------------------


# End infections plotting ventilation and adherency

intervention_effcs = [0.0,0.2,0.4] #,0.6]
interv_legend_label = [r'$0\%$ intervention efficiency',r'$20\%$ intervention efficiency',r'$40\%$ intervention efficiency'] #,r'$40\%$ intervention efficiency',r'$60\%$ intervention efficiency'] #,r'No intervention, schools $100\%$ occupation']

school_cap = 0.35

fraction_people_masked = [0.5,0.65,0.8,0.95,1.0]

ventilation_vals = [0.0,]

masks = ['cloth','surgical','N95']
masks_labels = ['Cloth','Surgical','N95']
masks_labels = dict(zip(masks,masks_labels))

states_ = ['S', 'E', 'I1', 'I2', 'I3', 'D', 'R']
df_list = []

inter_ = intervention_effcs[2]

for m, mask_ in enumerate(masks):
    for i, frac_mask_ in enumerate(fraction_people_masked):
        for j, vent_ in enumerate(ventilation_vals):

            res_read = load_results_ints('soln_cum',args.population,inter_,school_cap,mask_,frac_mask_,vent_,path=results_path)

            for itr_ in range(10):
                res_read_i = res_read['iter'] == itr_
                res_read_i = pd.DataFrame(res_read[res_read_i])
                end_cases = res_read_i['E'].iloc[-1]

                df_res_i = pd.DataFrame(columns=['iter','mask','frac_mask','interven_eff','ventilation','end_cases'])
                df_res_i['iter']         = [int(itr_)]
                df_res_i['mask']         = masks_labels[mask_]
                df_res_i['frac_mask']    = r'{}%'.format(int(frac_mask_*100))
                df_res_i['interven_eff'] = r'{}%'.format(int(inter_*100))
                df_res_i['ventilation']   = str(vent_)
                df_res_i['end_cases']      = end_cases*pop
                df_list.append(df_res_i)

df_final_E_v = pd.concat(df_list)
fig,ax = plt.subplots(1,1,figsize=(7, 6))
sns.pointplot(ax=ax, data=df_final_E_v, x='end_cases', y='frac_mask', hue='mask', linestyles='',dodge=0.3,palette='plasma',alpha=0.8)
ax.legend(bbox_to_anchor=(1.02,1)).set_title('')
plt.setp(ax.get_legend().get_texts(), fontsize='17') # for legend text
ax.set_xlabel(r'Infections per 10,000',fontsize=17)
ax.set_ylabel(r'Individuals wearing masks ($\%$)',fontsize=17)
ax.set_title(r'Total infections | schools at {}$\%$, low ventilation'.format(str(school_cap*100)),fontsize=17)
plt.xticks(size=16)
plt.yticks(size=16)
#plt.xlim([4850,6000])

save_path = os.path.join(figures_path,'point_plots','totalInfections_n_{}_schoolcap_{}_ventilation_{}_inter_{}.png'.format(str(pop),str(0.35),str(ventilation_vals[0]),str(inter_)))
plt.savefig(save_path,dpi=400, transparent=False, bbox_inches='tight', pad_inches=0.1 )


ventilation_vals = [15.0]

fraction_people_masked = [0.5,0.65,0.8,0.95,1.0]

masks = ['cloth','surgical','N95']
masks_labels = ['Cloth','Surgical','N95']
masks_labels = dict(zip(masks,masks_labels))

states_ = ['S', 'E', 'I1', 'I2', 'I3', 'D', 'R']
df_list = []

inter_ = intervention_effcs[2]

for m, mask_ in enumerate(masks):
    for i, frac_mask_ in enumerate(fraction_people_masked):
        for j, vent_ in enumerate(ventilation_vals):

            res_read = load_results_ints('soln_cum',args.population,inter_,school_cap,mask_,frac_mask_,vent_,path=results_path)

            for itr_ in range(10):
                res_read_i = res_read['iter'] == itr_
                res_read_i = pd.DataFrame(res_read[res_read_i])
                end_cases = res_read_i['E'].iloc[-1]

                df_res_i = pd.DataFrame(columns=['iter','mask','frac_mask','interven_eff','ventilation','end_cases'])
                df_res_i['iter']         = [int(itr_)]
                df_res_i['mask']         = masks_labels[mask_]
                df_res_i['frac_mask']    = r'{}%'.format(int(frac_mask_*100))
                df_res_i['interven_eff'] = r'{}%'.format(int(inter_*100))
                df_res_i['ventilation']   = str(vent_)
                df_res_i['end_cases']      = end_cases*pop
                df_list.append(df_res_i)

df_final_E_v = pd.concat(df_list)
fig,ax = plt.subplots(1,1,figsize=(7, 6))
sns.pointplot(ax=ax, data=df_final_E_v, x='end_cases', y='frac_mask', hue='mask', linestyles='',dodge=0.2,palette='plasma',alpha=0.8)
ax.legend(bbox_to_anchor=(1.02,1)).set_title('')
plt.setp(ax.get_legend().get_texts(), fontsize='17') # for legend text
ax.set_xlabel(r'Infections per 10,000',fontsize=17)
ax.set_ylabel(r'Individuals wearing masks ($\%$)',fontsize=17)
ax.set_title(r'Total infections | schools at {}$\%$, high ventilation'.format(str(school_cap*100)),fontsize=17)
plt.xticks(size=16)
plt.yticks(size=16)


save_path = os.path.join(figures_path,'point_plots','totalInfections_n_{}_schoolcap_{}_ventilation_{}_inter_{}.png'.format(str(pop),str(0.35),str(ventilation_vals[0]),str(inter_)))
plt.savefig(save_path,dpi=400, transparent=False, bbox_inches='tight', pad_inches=0.1 )
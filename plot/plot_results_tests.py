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
figures_path = config_data.loc['figures_test_dir'][1]
results_path = config_data.loc['results_test_dir'][1]
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

def load_results_ints(type_res,n,int_effec,schl_occup,layer,path=results_path):
    read_path = os.path.join(path,'{}_layerInt_{}_inter_{}_schoolcap_{}_{}.csv'.format(str(n),str(layer),str(int_effec),
                                                                           str(schl_occup),type_res))
    read_file = pd.read_csv(read_path)
    return read_file


### Read file

results_path = os.path.join(results_path,str(pop))


###------------------------------------------------------------------------------------------------------------------------------------------------------

### Bar plots

intervention_effcs = [0.0,0.2,0.4]
school_cap = 1.0 #,0.35]
layers_test = ['work','community','all']

df_list = []

for l, layer_ in enumerate(layers_test):
    for i, inter_ in enumerate(intervention_effcs):
        for j, schl_cap_ in enumerate(school_cap):

            res_read = load_results_ints('soln_cum',args.population,inter_,schl_cap_,layer_,results_path)

            layer_label = None
            if layer_ == 'work':
                layer_label = 'Intervention over work'
            elif layer_ == 'community':
                layer_label == 'Intervention over community'
            elif layer_ == 'all':
                layer_label == 'Intervention over-all'

            for itr_ in range(10):
                res_read_i = res_read['iter'] == itr_
                res_read_i = pd.DataFrame(res_read[res_read_i])
                end_cases = res_read_i['E'].iloc[-1]

                df_res_i = pd.DataFrame(columns=['iter','layer','interven_eff','end_cases'])
                df_res_i['iter']         = [int(itr_)]
                df_res_i['layer']        = layer_label
                df_res_i['interven_eff'] = r'{}$\%$'.format(int(inter_*100))
                df_res_i['end_cases']    = end_cases*pop
                df_list.append(df_res_i)

df_final_E = pd.concat(df_list)

fig,ax = plt.subplots(1,1,figsize=(7, 6))
sns.catplot(ax=ax, data=df_final_E, x='interven_eff', y='end_cases', hue='layer',alpha=0.5)
ax.legend(bbox_to_anchor=(1.02,1)).set_title('')
plt.setp(ax.get_legend().get_texts(), fontsize='17') # for legend text
ax.set_xlabel(r'Intervention efficiency ($\%$)',fontsize=17)
ax.set_ylabel(r'Infections per 10,000',fontsize=17)
ax.set_title(r'Total infections with schools at {}$\%$'.format(str(school_cap*100)),fontsize=17)
plt.xticks(size=17)
plt.yticks(size=17)

save_path = os.path.join(figures_path,'totalInfections_n_{}_schoolcap_{}_.png'.format(str(pop),str(0.35)))
plt.savefig(save_path,dpi=400, transparent=True, bbox_inches='tight', pad_inches=0.1 )


# Deaths

df_list = []

for l, layer_ in enumerate(layers_test):
    for i, inter_ in enumerate(intervention_effcs):
        for j, schl_cap_ in enumerate(school_cap):

            res_read = load_results_ints('soln_cum',args.population,inter_,schl_cap_,layer_,results_path)

            layer_label = None
            if layer_ == 'work':
                layer_label = 'Intervention over work'
            elif layer_ == 'community':
                layer_label == 'Intervention over community'
            elif layer_ == 'all':
                layer_label == 'Intervention over-all'

            for itr_ in range(10):
                res_read_i = res_read['iter'] == itr_
                res_read_i = pd.DataFrame(res_read[res_read_i])
                end_cases = res_read_i['D'].iloc[-1]

                df_res_i = pd.DataFrame(columns=['iter','layer','interven_eff','end_dead'])
                df_res_i['iter']         = [int(itr_)]
                df_res_i['layer']        = layer_label
                df_res_i['interven_eff'] = r'{}$\%$'.format(int(inter_*100))
                df_res_i['end_dead']    = end_dead*pop
                df_list.append(df_res_i)

df_final_D = pd.concat(df_list)

fig,ax = plt.subplots(1,1,figsize=(7, 6))
sns.catplot(ax=ax, data=df_final_D, x='interven_eff', y='end_dead', hue='layer',alpha=0.5)
ax.legend(bbox_to_anchor=(1.02,1)).set_title('')
plt.setp(ax.get_legend().get_texts(), fontsize='17') # for legend text
ax.set_xlabel(r'Intervention efficiency ($\%$)',fontsize=17)
ax.set_ylabel(r'Deaths per 10,000',fontsize=17)
ax.set_title(r'Total deaths with schools at {}$\%$'.format(str(school_cap*100)),fontsize=17)
plt.xticks(size=17)
plt.yticks(size=17)

save_path = os.path.join(figures_path,'totalDeaths_n_{}_schoolcap_{}_.png'.format(str(pop),str(0.35)))
plt.savefig(save_path,dpi=400, transparent=True, bbox_inches='tight', pad_inches=0.1 )
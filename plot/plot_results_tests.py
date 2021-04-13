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
school_cap = [1.0] #,0.35]
layers_test = ['work','community','all']
layers_labels = ['Intervention over work','Intervention over community','Intervention over-all']
layers_labels = dict(zip(layers_test,layers_labels))

df_list = []

for l, layer_ in enumerate(layers_test):
    for i, inter_ in enumerate(intervention_effcs):
        for j, schl_cap_ in enumerate(school_cap):

            res_read = load_results_ints('soln_cum',args.population,inter_,schl_cap_,layer_,results_path)

            for itr_ in range(10):
                res_read_i = res_read['iter'] == itr_
                res_read_i = pd.DataFrame(res_read[res_read_i])
                end_cases = res_read_i['E'].iloc[-1]

                df_res_i = pd.DataFrame(columns=['iter','Inter.Layer','interven_eff','end_cases'])
                df_res_i['iter']         = [int(itr_)]
                df_res_i['Inter.Layer']        = layers_labels[layer_]
                df_res_i['interven_eff'] = r'{}%'.format(int(inter_*100))
                df_res_i['end_cases']    = end_cases*pop
                df_list.append(df_res_i)

df_final_E = pd.concat(df_list)

fig,ax = plt.subplots(1,1,figsize=(9, 6))
sns.catplot(ax=ax, data=df_final_E, y='interven_eff', x='end_cases', hue='Inter.Layer',kind='bar',palette='winter',alpha=0.7,legend=False)
#ax.legend(bbox_to_anchor=(1.02,1)).set_title('')
plt.legend(bbox_to_anchor=(1.02,0.6),title='',frameon=False, fontsize=16)
#plt.setp(ax.get_legend().get_texts(), fontsize='17') # for legend text
plt.ylabel(r'Intervention efficiency ($\%$)',fontsize=17)
plt.xlabel(r'Infections per 10,000',fontsize=17)
plt.title(r'Total infections | schools at {}%'.format(str(int(school_cap[0]*100))),fontsize=17)
plt.xticks(size=16)
plt.yticks(size=16)

save_path = os.path.join(figures_path,'bar_plots','layersInter_totalInfections_n_{}_schoolcap_{}_.png'.format(str(pop),str(school_cap[0])))
plt.savefig(save_path,dpi=400, transparent=False, bbox_inches='tight', pad_inches=0.1 )


# Deaths

school_cap = [0.35] #,0.35]
layers_test = ['work','community','all']
layers_labels = ['Intervention over work','Intervention over community','Intervention over-all']
layers_labels = dict(zip(layers_test,layers_labels))

df_list = []

for l, layer_ in enumerate(layers_test):
    for i, inter_ in enumerate(intervention_effcs):
        for j, schl_cap_ in enumerate(school_cap):

            res_read = load_results_ints('soln_cum',args.population,inter_,schl_cap_,layer_,results_path)

            for itr_ in range(10):
                res_read_i = res_read['iter'] == itr_
                res_read_i = pd.DataFrame(res_read[res_read_i])
                end_dead = res_read_i['D'].iloc[-1]

                df_res_i = pd.DataFrame(columns=['iter','Inter.Layer','interven_eff','end_dead'])
                df_res_i['iter']         = [int(itr_)]
                df_res_i['Inter.Layer']        = layers_labels[layer_]
                df_res_i['interven_eff'] = r'{}%'.format(int(inter_*100))
                df_res_i['end_dead']    = end_dead*pop
                df_list.append(df_res_i)

df_final_D = pd.concat(df_list)

fig,ax = plt.subplots(1,1,figsize=(9, 6))
sns.catplot(ax=ax, data=df_final_D, y='interven_eff', x='end_dead', hue='Inter.Layer',kind='bar',palette='winter',alpha=0.7,legend=False)
#ax.legend(bbox_to_anchor=(1.02,1)).set_title('')
plt.legend(bbox_to_anchor=(1.02,0.6),title='',frameon=False, fontsize=16)
#plt.setp(ax.get_legend().get_texts(), fontsize='17') # for legend text
plt.ylabel(r'Intervention efficiency ($\%$)',fontsize=17)
plt.xlabel(r'Deaths per 10,000',fontsize=17)
plt.title(r'Total deaths | schools at {}%'.format(str(int(school_cap[0]*100))),fontsize=17)
plt.xticks(size=16)
plt.yticks(size=16)
plt.xlim([0,141])

save_path = os.path.join(figures_path,'bar_plots','layersInter_totalDeaths_n_{}_schoolcap_{}_.png'.format(str(pop),str(school_cap[0])))
plt.savefig(save_path,dpi=400, transparent=False, bbox_inches='tight', pad_inches=0.1 )

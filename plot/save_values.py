import sys
sys.path.append('../')

from matplotlib import figure
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math
import os
from tqdm import tqdm
from models import model

### Config folders

config_data = pd.read_csv('config.csv', sep=',', header=None, index_col=0)
figures_path = config_data.loc['figures_dir'][1]
results_test_path = config_data.loc['results_test_dir'][1]
results_old_path = config_data.loc['results_old_dir'][1]
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

def load_results_ints_1(type_res,n,int_effec,schl_occup,path=results_path):
    read_path = os.path.join(path,'{}_inter_{}_schoolcap_{}_{}.csv'.format(str(n),str(int_effec),
                                                                           str(schl_occup),type_res))
    read_file = pd.read_csv(read_path)
    return read_file

def load_results_ints_test(type_res,n,int_effec,schl_occup,layer,path=results_path):
    read_path = os.path.join(path,'{}_layerInt_{}_inter_{}_schoolcap_{}_{}.csv'.format(str(n),str(layer),str(int_effec),
                                                                           str(schl_occup),type_res))
    read_file = pd.read_csv(read_path)
    return read_file



### Read file

results_path = os.path.join(results_path,'intervention',str(pop))
results_test_path = os.path.join(results_test_path,str(pop))
results_old_path = os.path.join(results_old_path,'intervention',str(pop))


###------------------------------------------------------------------------------------------------------------------------------------------------------

# Final cases FIGURE 1

intervention_effcs = [0.0,0.2,0.4,0.6]
interv_legend_label = [r'$0\%$ intervention efficiency',r'$20\%$ intervention efficiency',r'$40\%$ intervention efficiency',r'$60\%$ intervention efficiency'] #,r'No intervention, schools $100\%$ occupation']
school_caps        = [0.0,0.15,0.25,0.35,0.55,1.0]
states_ = ['S', 'E', 'I1', 'I2', 'I3', 'D', 'R']
df_list = []
state_save = 'D'
for i, inter_ in enumerate(intervention_effcs):
    for j, cap_ in enumerate(school_caps):

        res_read = load_results_ints_1('soln_cum',args.population,inter_,cap_,path=results_old_path)

        for itr_ in range(10):
            res_read_i = res_read['iter'] == itr_
            res_read_i = pd.DataFrame(res_read[res_read_i])
            end_cases = res_read_i[state_save].iloc[-1]

            df_res_i = pd.DataFrame(columns=['iter','interven_eff','school_cap','end_cases'])
            df_res_i['iter']         = [int(itr_)]
            df_res_i['interven_eff'] = r'{}%'.format(int(inter_*100))
            df_res_i['school_cap']   = int(cap_*100)
            df_res_i['end_cases']      = end_cases*100
            df_list.append(df_res_i)

df_end_E = pd.concat(df_list)

list_interven_eff = [r'{}%'.format(int(int_*100)) for int_ in intervention_effcs]
df_list = []
for inter_eff in list_interven_eff:
    df_inter = df_end_E[df_end_E['interven_eff']==inter_eff]
    ci95_hi = []
    ci95_lo = []
    stats = df_inter.groupby(['school_cap'])['end_cases'].agg(['mean','count', 'std'])
    for i in stats.index:
        m, c, s = stats.loc[i]
        ci95_hi.append(m + 1.95*s/math.sqrt(c))
        ci95_lo.append(m - 1.95*s/math.sqrt(c))

    stats['ci95_hi'] = ci95_hi
    stats['ci95_lo'] = ci95_lo
    stats['interven_eff'] = [inter_eff]*len(ci95_hi)

    df_list.append(stats)

df_final_stats = pd.concat(df_list)
df_final_stats.to_excel('figure1_vals_{}.xlsx'.format(state_save))
###------------------------------------------------------------------------------------------------------------------------------------------------------

### Bar plots testes FIGURE 2

intervention_effcs = [0.0,0.2,0.4]
school_cap = [1.0] #,0.35]
layers_test = ['work','community','all']
layers_labels = ['Intervención sobre sitios de trabajo','Intervención sobre comunidad','Intervención completa']
layers_labels = dict(zip(layers_test,layers_labels))

df_list = []

for l, layer_ in enumerate(layers_test):
    for i, inter_ in enumerate(intervention_effcs):
        for j, schl_cap_ in enumerate(school_cap):

            res_read = load_results_ints_test('soln_cum',args.population,inter_,schl_cap_,layer_,results_test_path)

            for itr_ in range(10):
                res_read_i = res_read['iter'] == itr_
                res_read_i = pd.DataFrame(res_read[res_read_i])
                end_cases = res_read_i['E'].iloc[-1]

                df_res_i = pd.DataFrame(columns=['iter','Inter.Layer','interven_eff','end_cases'])
                df_res_i['iter']         = [int(itr_)]
                df_res_i['School.Cap']         = r'{}%'.format(int(schl_cap_*100))
                df_res_i['Inter.Layer']        = layers_labels[layer_]
                df_res_i['interven_eff'] = r'{}%'.format(int(inter_*100))
                df_res_i['end_cases']    = end_cases*100
                df_list.append(df_res_i)

df_final_E = pd.concat(df_list)

list_interven_eff = [r'{}%'.format(int(int_*100)) for int_ in intervention_effcs]
df_list = []
for inter_eff in list_interven_eff:
    df_inter = df_final_E[df_final_E['interven_eff']==inter_eff]
    ci95_hi = []
    ci95_lo = []
    stats = df_inter.groupby(['Inter.Layer'])['end_cases'].agg(['mean','count', 'std'])
    for i in stats.index:
        m, c, s = stats.loc[i]
        ci95_hi.append(m + 1.95*s/math.sqrt(c))
        ci95_lo.append(m - 1.95*s/math.sqrt(c))

    stats['ci95_hi'] = ci95_hi
    stats['ci95_lo'] = ci95_lo
    stats['interven_eff'] = [inter_eff]*len(ci95_hi)

    df_list.append(stats)

df_final_stats = pd.concat(df_list)
df_final_stats.to_csv('figure2_vals_{}.csv'.format(school_cap[0]))
###------------------------------------------------------------------------------------------------------------------------------------------------------

### Bar plots

# End infections plotting ventilation and mask FIGURE 3

intervention_effcs = [0.0,0.2,0.4] #,0.6]
interv_legend_label = [r'$0\%$ intervention efficiency',r'$20\%$ intervention efficiency',r'$40\%$ intervention efficiency'] #,r'$40\%$ intervention efficiency',r'$60\%$ intervention efficiency'] #,r'No intervention, schools $100\%$ occupation']

school_cap = 0.35

fraction_people_masked = 1.0

ventilation_vals = [0.0,5.0,8.0,15.0]
ventilation_labels = ['Cero','Baja','Media','Alta']
ventilation_labels = dict(zip(ventilation_vals,ventilation_labels))

masks = ['cloth','surgical','N95']
masks_labels = {'cloth':'Tela','surgical':'Quirúrgicos','N95':'N95'}

states_ = ['S', 'E', 'I1', 'I2', 'I3', 'D', 'R']
df_list = []

inter_ = intervention_effcs[2]
for m, mask_ in enumerate(masks):
    for j, vent_ in enumerate(ventilation_vals):

        res_read = load_results_ints('soln_cum',args.population,inter_,school_cap,mask_,fraction_people_masked,vent_,path=results_path)

        for itr_ in range(10):
            res_read_i = res_read['iter'] == itr_
            res_read_i = pd.DataFrame(res_read[res_read_i])
            end_cases = res_read_i['E'].iloc[-1]

            df_res_i = pd.DataFrame(columns=['iter','Tacapobas','interven_eff','ventilation','end_cases'])
            df_res_i['iter']         = [int(itr_)]
            df_res_i['Tacapobas']         = str(masks_labels[mask_])
            df_res_i['interven_eff'] = r'{}%'.format(int(inter_*100))
            df_res_i['ventilation']   = ventilation_labels[vent_]
            df_res_i['end_cases']      = end_cases*100
            df_list.append(df_res_i)

df_final_E = pd.concat(df_list)

ventilation_labels_s = ['Cero','Baja','Media','Alta']
df_list = []
for vent_vals in ventilation_labels_s:
    df_inter = df_final_E[df_final_E['ventilation']==vent_vals]
    ci95_hi = []
    ci95_lo = []
    stats = df_inter.groupby(['Tacapobas'])['end_cases'].agg(['mean','count', 'std'])
    for i in stats.index:
        m, c, s = stats.loc[i]
        ci95_hi.append(m + 1.95*s/math.sqrt(c))
        ci95_lo.append(m - 1.95*s/math.sqrt(c))

    stats['ci95_hi'] = ci95_hi
    stats['ci95_lo'] = ci95_lo
    stats['ventilation'] = [vent_vals]*len(ci95_hi)
    stats['interven_eff'] = [str(inter_*100)+'%']*len(ci95_hi)

    df_list.append(stats)

df_final_stats = pd.concat(df_list)
df_final_stats.to_csv('figure3_vals_{}.csv'.format(inter_))

###------------------------------------------------------------------------------------------------------------------------------------------------------

### Point plots

# End infections plotting ventilation and adherency

intervention_effcs = [0.0,0.2,0.4] #,0.6]
interv_legend_label = [r'$0\%$ intervention efficiency',r'$20\%$ intervention efficiency',r'$40\%$ intervention efficiency'] #,r'$40\%$ intervention efficiency',r'$60\%$ intervention efficiency'] #,r'No intervention, schools $100\%$ occupation']

school_cap = 0.35

fraction_people_masked = [1.0]

ventilation_vals = [0.0]

masks = ['cloth','surgical','N95']
masks_labels = ['Tela','Quirúrgico','N95']
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

                df_res_i = pd.DataFrame(columns=['iter','Tacapobas','frac_mask','interven_eff','ventilation','end_cases'])
                df_res_i['iter']         = [int(itr_)]
                df_res_i['Tacapobas']         = masks_labels[mask_]
                df_res_i['frac_mask']    = r'{}%'.format(int(frac_mask_*100))
                df_res_i['interven_eff'] = r'{}%'.format(int(inter_*100))
                df_res_i['ventilation']   = str(vent_)
                df_res_i['end_cases']      = end_cases*100
                df_list.append(df_res_i)

df_final_E_v = pd.concat(df_list)

df_list_s = []
for vent_vals in ventilation_vals:
    df_inter = df_final_E_v[df_final_E_v['ventilation']==str(vent_vals)]
    ci95_hi = []
    ci95_lo = []
    stats = df_inter.groupby(['Tacapobas'])['end_cases'].agg(['mean','count', 'std'])
    for i in stats.index:
        m, c, s = stats.loc[i]
        ci95_hi.append(m + 1.95*s/math.sqrt(c))
        ci95_lo.append(m - 1.95*s/math.sqrt(c))

    stats['ci95_hi'] = ci95_hi
    stats['ci95_lo'] = ci95_lo
    stats['ventilation'] = [vent_vals]*len(ci95_hi)
    stats['interven_eff'] = [str(inter_*100)+'%']*len(ci95_hi)

    df_list_s.append(stats)

df_final_stats = pd.concat(df_list_s)
df_final_stats.to_csv('figure4_vals_vent_{}.csv'.format(ventilation_vals[0]))
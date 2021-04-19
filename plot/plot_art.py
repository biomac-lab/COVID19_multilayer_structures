import sys
sys.path.append('../')

from matplotlib import figure
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.collections import PatchCollection
from matplotlib.colors import ListedColormap
import os
from tqdm import tqdm

### Config folders

config_data = pd.read_csv('config.csv', sep=',', header=None, index_col=0)
figures_path = config_data.loc['figures_dir_art'][1]
results_path = config_data.loc['results_dir'][1]
results_test_path = config_data.loc['results_test_dir'][1]
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

def load_results_ints_test(type_res,n,int_effec,schl_occup,layer,path=results_path):
    read_path = os.path.join(path,str(n),'{}_layerInt_{}_inter_{}_schoolcap_{}_{}.csv'.format(str(n),str(layer),str(int_effec),
                                                                           str(schl_occup),type_res))
    read_file = pd.read_csv(read_path)
    return read_file


### Read file

results_path = os.path.join(results_path,'intervention',str(pop))

###------------------------------------------------------------------------------------------------------------------------------------------------------
### Plot proportional areas (each mask type) for each level of ventilation


def nested_circles(data, labels=None, c=None, ax=None,
                   cmap=None, norm=None, textkw={}):
    ax = ax or plt.gca()
    data = np.array(data)
    R = np.sqrt(data/data.max())
    p = [plt.Circle((0,r), radius=r) for r in R[::-1]]
    arr = data[::-1] if c is None else np.array(c[::-1])
    col = PatchCollection(p, cmap=cmap, norm=norm, array=arr)

    ax.add_collection(col)
    ax.axis("off")
    ax.set_aspect("equal")
    ax.autoscale()

    if labels is not None:
        kw = dict(color="k", va="center", ha="center")
        kw.update(textkw)
        ax.text(0, R[0], labels[0], **kw)
        for i in range(1, len(R)):
            ax.text(0, R[i]+R[i-1], labels[i], **kw)
    return col

# from pylab import *
# cmap = cm.get_cmap('gist_heat_r', 5)    # PiYG
# colors_plt = []
# for i in range(cmap.N):
#     rgba = cmap(i)
#     # rgb2hex accepts rgb or rgba
#     colors_plt.append(matplotlib.colors.rgb2hex(rgba))
# colors_plt = colors_plt[1:4]

# def plot_examples(cms):
#     """
#     helper function to plot two colormaps
#     """
#     np.random.seed(19680801)
#     data = np.random.randn(30, 30)

#     fig, axs = plt.subplots(1, 2, figsize=(6, 3), constrained_layout=True)
#     for [ax, cmap] in zip(axs, cms):
#         psm = ax.pcolormesh(data, cmap=cmap, rasterized=True, vmin=-4, vmax=4)
#         fig.colorbar(psm, ax=ax)
#     plt.show()


# viridisBig = cm.get_cmap('Reds_r', 512)
# newcmp = ListedColormap(viridisBig(np.linspace(0.95, 0.5, 256)))


# school_cap = 0.35
# fraction_people_masked = 1.0
# ventilation_vals = 0.0
# inter_ = 0.4

# masks = ['cloth','surgical','N95']
# masks_labels = ['Cloth','Surgical','N95']
# masks_labels = dict(zip(masks,masks_labels))

# df_list = []


# for m, mask_ in enumerate(masks):
#     res_read = load_results_ints('soln_cum',args.population,inter_,school_cap,mask_,fraction_people_masked,ventilation_vals,path=results_path)

#     for itr_ in range(10):
#         res_read_i = res_read['iter'] == itr_
#         res_read_i = pd.DataFrame(res_read[res_read_i])
#         end_cases = res_read_i['E'].iloc[-1]

#         df_res_i = pd.DataFrame(columns=['iter','mask','frac_mask','interven_eff','ventilation','end_cases'])
#         df_res_i['iter']         = [int(itr_)]
#         df_res_i['mask']         = masks_labels[mask_]
#         df_res_i['frac_mask']    = r'{}%'.format(int(fraction_people_masked*100))
#         df_res_i['interven_eff'] = r'{}%'.format(int(inter_*100))
#         df_res_i['ventilation']   = str(ventilation_vals)
#         df_res_i['end_cases']      = end_cases*100
#         df_list.append(df_res_i)

# df_final_E_lowVent = pd.concat(df_list)
# df_final_E_lowVent_meds = df_final_E_lowVent.groupby('mask').median().reset_index()
# percentagesData_E_lowVent_mends = list(df_final_E_lowVent_meds['end_cases'])
# percentagesLabels_E_lowVent_mends = [r'{:.2f}%'.format(end_cases) for end_cases in df_final_E_lowVent_meds['end_cases']]
# nested_circles(percentagesData_E_lowVent_mends,labels=percentagesLabels_E_lowVent_mends,cmap='copper',textkw=dict(fontsize=14))
# plt.show()


# test_vals = [8.420,100-8.420]
# test_labels = list("AB")
# nested_circles(test_vals, labels=test_labels, cmap="copper", textkw=dict(fontsize=14))
# plt.show()

# fig,ax = plt.subplots(1,1,figsize=(7, 6))
# sns.pointplot(ax=ax, data=df_final_E_v, x='end_cases', y='frac_mask', hue='mask', linestyles='',dodge=0.3,palette='plasma',alpha=0.8)
# ax.legend(bbox_to_anchor=(1.02,1)).set_title('')
# plt.setp(ax.get_legend().get_texts(), fontsize='17') # for legend text
# ax.set_xlabel(r'Infections per 10,000',fontsize=17)
# ax.set_ylabel(r'Individuals wearing masks ($\%$)',fontsize=17)
# ax.set_title(r'Total infections | schools at {}$\%$, low ventilation'.format(str(school_cap*100)),fontsize=17)
# plt.xticks(size=16)
# plt.yticks(size=16)
# #plt.xlim([4850,6000])

# save_path = os.path.join(figures_path,'point_plots','totalInfections_n_{}_schoolcap_{}_ventilation_{}_inter_{}.png'.format(str(pop),str(0.35),str(ventilation_vals[0]),str(inter_)))
# plt.savefig(save_path,dpi=400, transparent=False, bbox_inches='tight', pad_inches=0.1 )


# school_cap = 0.35
# fraction_people_masked = 1.0
# ventilation_vals = 15.0
# inter_ = 0.4

# masks = ['cloth','surgical','N95']
# masks_labels = ['Cloth','Surgical','N95']
# masks_labels = dict(zip(masks,masks_labels))

# df_list = []

# for m, mask_ in enumerate(masks):
#     res_read = load_results_ints('soln_cum',args.population,inter_,school_cap,mask_,fraction_people_masked,ventilation_vals,path=results_path)

#     for itr_ in range(10):
#         res_read_i = res_read['iter'] == itr_
#         res_read_i = pd.DataFrame(res_read[res_read_i])
#         end_cases = res_read_i['E'].iloc[-1]

#         df_res_i = pd.DataFrame(columns=['iter','mask','frac_mask','interven_eff','ventilation','end_cases'])
#         df_res_i['iter']         = [int(itr_)]
#         df_res_i['mask']         = masks_labels[mask_]
#         df_res_i['frac_mask']    = r'{}%'.format(int(fraction_people_masked*100))
#         df_res_i['interven_eff'] = r'{}%'.format(int(inter_*100))
#         df_res_i['ventilation']   = str(ventilation_vals)
#         df_res_i['end_cases']      = end_cases*100
#         df_list.append(df_res_i)

# df_final_E_highVent = pd.concat(df_list)
# df_final_E_highVent_meds = df_final_E_highVent.groupby('mask').median().reset_index()
# percentagesData_E_lowVent_mends = list(df_final_E_highVent_meds['end_cases'])
# percentagesLabels_E_lowVent_mends = [r'{:.2f}%'.format(end_cases) for end_cases in df_final_E_highVent_meds['end_cases']]
# nested_circles(percentagesData_E_lowVent_mends,labels=percentagesLabels_E_lowVent_mends,cmap='copper',textkw=dict(fontsize=14))
# plt.show()


# test_vals = [30.035,100-30.035]
# test_labels = list("AB")
# nested_circles(test_vals, labels=test_labels, cmap="copper", textkw=dict(fontsize=14))
# plt.show()

###------------------------------------------------------------------------------------------------------------------------------------------------------

### Bar plots testes

intervention_effcs = [0.0,0.2,0.4]
school_cap = [0.35] #,0.35]
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
                df_res_i['Inter.Layer']        = layers_labels[layer_]
                df_res_i['interven_eff'] = r'{}%'.format(int(inter_*100))
                df_res_i['end_cases']    = end_cases*100
                df_list.append(df_res_i)

df_final_E = pd.concat(df_list)

fig,ax = plt.subplots(1,1,figsize=(9, 6))
sns.catplot(ax=ax, data=df_final_E, y='interven_eff', x='end_cases', hue='Inter.Layer',kind='bar',palette='Blues',alpha=0.7,legend=False)
#ax.legend(bbox_to_anchor=(1.02,1)).set_title('')
plt.legend(bbox_to_anchor=(1.02,0.6),title='',frameon=False, fontsize=16)
#plt.setp(ax.get_legend().get_texts(), fontsize='17') # for legend text
plt.ylabel(r'Efficiencia de intervención, ($\%$)',fontsize=17)
plt.xlabel(r'% Infectados',fontsize=17)
plt.title(r'Infecciones totales | Colegios al {}%'.format(str(int(school_cap[0]*100))),fontsize=17)
plt.xticks(size=16)
plt.yticks(size=16)

save_path = os.path.join(figures_path,'bar_plots','layersInter_totalInfections_n_{}_schoolcap_{}_.png'.format(str(pop),str(school_cap[0])))
plt.savefig(save_path,dpi=400, transparent=False, bbox_inches='tight', pad_inches=0.1 )



###------------------------------------------------------------------------------------------------------------------------------------------------------

### Bar plots

# End infections plotting ventilation and mask

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

inter_ = intervention_effcs[0]
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

plt.figure(figsize=(7,6))
sns.catplot(data=df_final_E, x='ventilation', y='end_cases', hue='Tacapobas',kind='bar',palette='Reds_r',alpha=0.8)
#ax.legend(bbox_to_anchor=(1.02,1)).set_title('')
#plt.setp(ax.get_legend().get_texts(), fontsize='17') # for legend text
plt.xlabel('Ventilación',fontsize=17)
plt.ylabel(r'% Infectados',fontsize=17)
plt.title(r'Infecciones totales | colegios {}$\%$, intervención {}$\%$'.format(str(int(school_cap*100)),str(int(inter_*100))),fontsize=17)
plt.xticks(size=16)
plt.yticks(size=16)
plt.ylim([0,101])
#plt.show()
save_path = os.path.join(figures_path,'bar_plots','totalInfections_n_{}_inter_{}_schoolcap_{}_.png'.format(str(pop),str(inter_),str(0.35)))
plt.savefig(save_path,dpi=400, transparent=False, bbox_inches='tight', pad_inches=0.1 )

# End deaths plotting ventilation and mask

inter_ = intervention_effcs[2]
for m, mask_ in enumerate(masks):
    for j, vent_ in enumerate(ventilation_vals):

        res_read = load_results_ints('soln_cum',args.population,inter_,school_cap,mask_,fraction_people_masked,vent_,path=results_path)

        for itr_ in range(10):
            res_read_i = res_read['iter'] == itr_
            res_read_i = pd.DataFrame(res_read[res_read_i])
            end_dead = res_read_i['D'].iloc[-1]

            df_res_i = pd.DataFrame(columns=['iter','Mask','interven_eff','ventilation','end_dead'])
            df_res_i['iter']         = [int(itr_)]
            df_res_i['Mask']         = str(masks_labels[mask_])
            df_res_i['interven_eff'] = r'{}%'.format(int(inter_*100))
            df_res_i['ventilation']   = ventilation_labels[vent_]
            df_res_i['end_dead']      = end_dead*100
            df_list.append(df_res_i)

df_final_D = pd.concat(df_list)

plt.figure(figsize=(7,6))
sns.catplot(data=df_final_D, x='ventilation', y='end_dead', hue='Mask',kind='bar',palette='plasma',alpha=0.8)
#ax.legend(bbox_to_anchor=(1.02,1)).set_title('')
#plt.setp(ax.get_legend().get_texts(), fontsize='17') # for legend text
plt.xlabel('Ventilation',fontsize=17)
plt.ylabel(r'Deaths per 10,000',fontsize=17)
plt.title(r'Total deaths | schools {}$\%$, intervention {}$\%$'.format(str(int(school_cap*100)),str(int(inter_*100))),fontsize=17)
plt.xticks(size=16)
plt.yticks(size=16)
plt.ylim([0,125])
#plt.show()
save_path = os.path.join(figures_path,'bar_plots','totalDeaths_n_{}_inter_{}_schoolcap_{}_.png'.format(str(pop),str(inter_),str(0.35)))
plt.savefig(save_path,dpi=400, transparent=False, bbox_inches='tight', pad_inches=0.1 )

###------------------------------------------------------------------------------------------------------------------------------------------------------

### Point plots

# End infections plotting ventilation and adherency

intervention_effcs = [0.0,0.2,0.4] #,0.6]
interv_legend_label = [r'$0\%$ intervention efficiency',r'$20\%$ intervention efficiency',r'$40\%$ intervention efficiency'] #,r'$40\%$ intervention efficiency',r'$60\%$ intervention efficiency'] #,r'No intervention, schools $100\%$ occupation']

school_cap = 0.35

fraction_people_masked = [0.5,0.65,0.8,0.95,1.0]

ventilation_vals = [0.0]

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

###------------------------------------------------------------------------------------------------------------------------------------------------------

### Bar plots

# End infections plotting ventilation and mask

intervention_effcs = [0.0,0.2,0.4] #,0.6]
interv_legend_label = [r'$0\%$ intervention efficiency',r'$20\%$ intervention efficiency',r'$40\%$ intervention efficiency'] #,r'$40\%$ intervention efficiency',r'$60\%$ intervention efficiency'] #,r'No intervention, schools $100\%$ occupation']

school_cap = 0.35

fraction_people_masked = 1.0

ventilation_vals = [0.0,5.0,8.0,15.0]
ventilation_labels_ = ['Cero','Low','Medium','High']
ventilation_labels = dict(zip(ventilation_vals,ventilation_labels_))

masks = ['cloth','surgical','N95']
masks_labels = {'cloth':'Cloth','surgical':'Surgical','N95':'N95'}

states_ = ['S', 'E', 'I1', 'I2', 'I3', 'D', 'R']
df_list = []

inter_ = intervention_effcs[0]
for m, mask_ in enumerate(masks):
    for j, vent_ in enumerate(ventilation_vals):

        res_read = load_results_ints('soln_cum',args.population,inter_,school_cap,mask_,fraction_people_masked,vent_,path=results_path)

        for itr_ in range(10):
            res_read_i = res_read['iter'] == itr_
            res_read_i = pd.DataFrame(res_read[res_read_i])
            end_cases = res_read_i['E'].iloc[-1]

            df_res_i = pd.DataFrame(columns=['iter','Mask','interven_eff','ventilation','end_cases'])
            df_res_i['iter']         = [int(itr_)]
            df_res_i['Mask']         = str(masks_labels[mask_])
            df_res_i['interven_eff'] = r'{}%'.format(int(inter_*100))
            df_res_i['ventilation']   = ventilation_labels[vent_]
            df_res_i['end_cases']      = end_cases*100
            df_list.append(df_res_i)

df_final_E = pd.concat(df_list)

# df_final_ventVals_E = df_final_E.groupby('ventilation').mean().reset_index()

df_list_vent = []
for vent_ in ventilation_labels_:
    df_final_vent_ = df_final_E[df_final_E['ventilation']==vent_]
    df_mask_means  = df_final_vent_.groupby('ventilation').mean().reset_index()
    df_list_vent.append(df_mask_means)

df_final_ventVals_E = pd.concat(df_list_vent)


plt.figure(figsize=(7,6))
sns.catplot(data=df_final_ventVals_E, x='ventilation', y='end_cases', kind='bar',palette='plasma',alpha=0.8)
#ax.legend(bbox_to_anchor=(1.02,1)).set_title('')
#plt.setp(ax.get_legend().get_texts(), fontsize='17') # for legend text
plt.xlabel('Ventilation',fontsize=17)
plt.ylabel(r'Infections ($\%$)',fontsize=17)
plt.title(r'Total infections | schools {}$\%$, intervention {}$\%$'.format(str(int(school_cap*100)),str(int(inter_*100))),fontsize=17)
plt.xticks(size=16)
plt.yticks(size=16)
plt.ylim([0,55])
#plt.show()
save_path = os.path.join(figures_path,'bar_plots','AVRGMAKS_totalInfections_n_{}_inter_{}_schoolcap_{}_.png'.format(str(pop),str(inter_),str(0.35)))
plt.savefig(save_path,dpi=400, transparent=False, bbox_inches='tight', pad_inches=0.1 )


###------------------------------------------------------------------------------------------------------------------------------------------------------

### Point plots

# End infections plotting ventilation and mask

intervention_effcs = [0.0,0.2,0.4,0.6]
interv_legend_label = [r'$0\%$ intervention efficiency',r'$20\%$ intervention efficiency',r'$40\%$ intervention efficiency',r'$40\%$ intervention efficiency',r'$60\%$ intervention efficiency'] #,r'No intervention, schools $100\%$ occupation']

school_caps        = [0.0,0.15,0.25,0.35,0.55]

ventilation_vals = [0.0,15.0]
ventilation_labels_ = ['Cero','Low','Medium','High']
ventilation_labels = dict(zip(ventilation_vals,ventilation_labels_))

fraction_people_masked = 0.8

mask = 'surgical'

# cases
df_list = []
state_save = 'E'
for i, inter_ in enumerate(intervention_effcs):
    for j, cap_ in enumerate(school_caps):

        res_read = load_results_ints('soln_cum',args.population,inter_,cap_,mask,fraction_people_masked,0.0,path=results_path)

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

fig,ax = plt.subplots(1,1,figsize=(7, 6))
sns.pointplot(ax=ax, data=df_end_E, x='school_cap', y='end_cases', hue='interven_eff', linestyles='',palette='viridis',alpha=0.5)
ax.legend(bbox_to_anchor=(1.02,1)).set_title('')
plt.setp(ax.get_legend().get_texts(), fontsize='17') # for legend text
ax.set_xlabel(r'School capacity ($\%$)',fontsize=17)
ax.set_ylabel(r'Infections ($\%$)',fontsize=17)
ax.set_title(r'Total infections | 80% using {} masks, low ventilation'.format(mask),fontsize=17)
plt.xticks(size=17)
plt.yticks(size=17)
# plt.show()
save_path = os.path.join(figures_path,'point_plots','LV_totalInfections_08_mask_{}_n_{}.png'.format(mask,str(pop)))
plt.savefig(save_path,dpi=400, transparent=True, bbox_inches='tight', pad_inches=0.1 )

# deaths
df_list = []
state_save = 'D'
for i, inter_ in enumerate(intervention_effcs):
    for j, cap_ in enumerate(school_caps):

        res_read = load_results_ints('soln_cum',args.population,inter_,cap_,mask,fraction_people_masked,0.0,path=results_path)

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

df_end_D = pd.concat(df_list)

fig,ax = plt.subplots(1,1,figsize=(7, 6))
sns.pointplot(ax=ax, data=df_end_D, x='school_cap', y='end_cases', hue='interven_eff', linestyles='',palette='viridis',alpha=0.5)
ax.legend(bbox_to_anchor=(1.02,1)).set_title('')
plt.setp(ax.get_legend().get_texts(), fontsize='17') # for legend text
ax.set_xlabel(r'School capacity ($\%$)',fontsize=17)
ax.set_ylabel(r'Deaths ($\%$)',fontsize=17)
ax.set_title(r'Total deaths | 80% using {} masks, low ventilation'.format(mask),fontsize=17)
plt.xticks(size=17)
plt.yticks(size=17)
# plt.show()
save_path = os.path.join(figures_path,'point_plots','LV_totalDeaths_08_mask_{}_n_{}.png'.format(mask,str(pop)))
plt.savefig(save_path,dpi=400, transparent=True, bbox_inches='tight', pad_inches=0.1 )

################
# End infections plotting high and mask

intervention_effcs = [0.0,0.2,0.4,0.6]
interv_legend_label = [r'$0\%$ intervention efficiency',r'$20\%$ intervention efficiency',r'$40\%$ intervention efficiency',r'$40\%$ intervention efficiency',r'$60\%$ intervention efficiency'] #,r'No intervention, schools $100\%$ occupation']

school_caps        = [0.0,0.15,0.25,0.35,0.55]

ventilation_vals = [0.0,15.0]
ventilation_labels_ = ['Cero','Low','Medium','High']
ventilation_labels = dict(zip(ventilation_vals,ventilation_labels_))

fraction_people_masked = 0.8

# cases
df_list = []
state_save = 'E'
for i, inter_ in enumerate(intervention_effcs):
    for j, cap_ in enumerate(school_caps):

        res_read = load_results_ints('soln_cum',args.population,inter_,cap_,mask,fraction_people_masked,15.0,path=results_path)

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

fig,ax = plt.subplots(1,1,figsize=(7, 6))
sns.pointplot(ax=ax, data=df_end_E, x='school_cap', y='end_cases', hue='interven_eff', linestyles='',palette='viridis',alpha=0.5)
ax.legend(bbox_to_anchor=(1.02,1)).set_title('')
plt.setp(ax.get_legend().get_texts(), fontsize='17') # for legend text
ax.set_xlabel(r'School capacity ($\%$)',fontsize=17)
ax.set_ylabel(r'Infections ($\%$)',fontsize=17)
ax.set_title(r'Total infections | 80% using {} masks, high ventilation'.format(mask),fontsize=17)
plt.xticks(size=17)
plt.yticks(size=17)
# plt.show()
save_path = os.path.join(figures_path,'point_plots','HV_totalInfections_08_mask_{}_n_{}.png'.format(mask,str(pop)))
plt.savefig(save_path,dpi=400, transparent=True, bbox_inches='tight', pad_inches=0.1 )

# deaths
df_list = []
state_save = 'D'
for i, inter_ in enumerate(intervention_effcs):
    for j, cap_ in enumerate(school_caps):

        res_read = load_results_ints('soln_cum',args.population,inter_,cap_,mask,fraction_people_masked,15.0,path=results_path)

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

df_end_D = pd.concat(df_list)

fig,ax = plt.subplots(1,1,figsize=(7, 6))
sns.pointplot(ax=ax, data=df_end_D, x='school_cap', y='end_cases', hue='interven_eff', linestyles='',palette='viridis',alpha=0.5)
ax.legend(bbox_to_anchor=(1.02,1)).set_title('')
plt.setp(ax.get_legend().get_texts(), fontsize='17') # for legend text
2
# plt.show()
save_path = os.path.join(figures_path,'point_plots','HV_totalDeaths_08_mask_{}_n_{}.png'.format(mask,str(pop)))
plt.savefig(save_path,dpi=400, transparent=True, bbox_inches='tight', pad_inches=0.1 )
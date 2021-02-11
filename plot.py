from matplotlib import figure
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from models import model

config_data = pd.read_csv('configlin.csv', sep=',', header=None, index_col=0)
figures_path = config_data.loc['figures_dir'][1]
results_path = config_data.loc['results_dir'][1]
ages_data_path = config_data.loc['bogota_age_data_dir'][1]
houses_data_path = config_data.loc['bogota_houses_data_dir'][1]


import argparse

parser = argparse.ArgumentParser(description='Dynamics visualization.')

parser.add_argument('--population', default=100000, type=int,
                    help='Speficy the number of individials')
parser.add_argument('--type_sim', default='intervention', type=str,
                    help='Speficy the type of simulation to plot')
parser.add_argument('--intervention', default=0.6, type=float,
                    help='Intervention efficiancy')
parser.add_argument('--school_occupation', default=0.35, type=float,
                    help='Percentage of occupation at classrooms over intervention')

args = parser.parse_args()

number_nodes = args.population
pop = number_nodes

results_path = os.path.join(results_path,args.type_sim,str(pop))


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


### Plot new cases

intervention_effcs = [0.2,0.4,0.6,1.0]
interv_legend_label = [r'$20\%$ intervention efficiency',r'$40\%$ intervention efficiency',r'$60\%$ intervention efficiency',r'No intervention, schools $100\%$ occupation']
interv_color_label = ['tab:red','tab:purple','tab:orange','k']

states_ = ['S', 'E', 'I1', 'I2', 'I3', 'D', 'R']

# fig, ax = plt.subplots(1,2,figsize=(14, 4))
# for i, inter_ in tqdm(enumerate(intervention_effcs),total=len(intervention_effcs)):
#     # read results
#     if inter_ < 1.0:
#         res_read = load_results_ints('soln',args.population,inter_,args.school_occupation)
#         res_mean = res_read.groupby('tvec').mean(); res_mean = res_mean.reset_index()
#         res_loCI = res_read.groupby('tvec').quantile(0.05); res_loCI = res_loCI.reset_index()
#         res_upCI = res_read.groupby('tvec').quantile(0.95); res_upCI = res_upCI.reset_index()
#     # read results with no intervention
#     elif inter_ == 1.0:
#         res_read_ni = load_results_dyn('soln',os.path.join('results','no_intervention',str(pop),))
#         res_mean = res_read_ni.groupby('tvec').mean(); res_mean = res_mean.reset_index()
#         res_loCI = res_read_ni.groupby('tvec').quantile(0.05); res_loCI = res_loCI.reset_index()
#         res_upCI = res_read_ni.groupby('tvec').quantile(0.95); res_upCI = res_upCI.reset_index()


#     plt.subplot(121)
#     plt.plot(res_mean['tvec'],res_mean['E']*100,color=interv_color_label[i])
#     plt.legend(interv_legend_label,frameon=False,framealpha=0.0,bbox_to_anchor=(1,1), loc="best")
#     plt.gca().set_prop_cycle(None)
#     plt.fill_between(res_mean['tvec'],res_loCI['E']*100,res_upCI['E']*100,color=interv_color_label[i],alpha=0.3)
#     #plt.plot([20,20],[0,100],'k--',alpha=0.2)1.0
#     plt.xlim([0,max(res_mean['tvec'])])
#     plt.ylim([0,0.1*100])
#     plt.xticks(size=12)
#     plt.yticks(size=12)
#     plt.xlabel("Time (days)",size=12)
#     plt.ylabel(r"$\%$ new cases per 100,000 ind",size=12)
#     if args.type_sim == 'intervention':
#         plt.title(r'New cases with schools opening ${:.2f}\%$ occupation'.format(args.school_occupation*100))
#     elif args.type_sim == 'school_alternancy':
#         plt.title(r'New cases with schools alterning ${:.2f}\%$ occupation'.format(args.school_occupation*100))

#     plt.subplot(122)
#     plt.plot(res_mean['tvec'],res_mean['E']*100,color=interv_color_label[i])
#     plt.gca().set_prop_cycle(None)
#     plt.fill_between(res_mean['tvec'],res_loCI['E']*100,res_upCI['E']*100,color=interv_color_label[i],alpha=0.3)
#     #plt.plot([20,20],[0,100],'k--',alpha=0.2)
#     plt.xlim([0,max(res_mean['tvec'])])
#     plt.ylim([1/pop*100,1*100])
#     plt.xticks(size=12)
#     plt.yticks(size=12)
#     plt.xlabel("Time (days)",size=12)
#     plt.ylabel(r"$\%$ new cases per 100,000 ind",size=12)
#     if args.type_sim == 'intervention':
#         plt.title(r'New cases with schools opening ${:.2f}\%$ occupation'.format(args.school_occupation*100))
#     elif args.type_sim == 'school_alternancy':
#         plt.title(r'New cases with schools alterning ${:.2f}\%$ occupation'.format(args.school_occupation*100))
#     plt.semilogy()
#     plt.tight_layout()


# if not os.path.isdir( os.path.join(figures_path,args.type_sim) ):
#     os.makedirs( os.path.join(figures_path,args.type_sim) )

# save_path = os.path.join(figures_path,args.type_sim)

# plt.savefig(os.path.join(figures_path,'{}_dynamics_schoolcap_{}_n_{}.png'.format(args.type_sim,args.school_occupation,str(pop))),
#             dpi=400, transparent=True, bbox_inches='tight', pad_inches=0.1 )

# plt.show()


### Plot dayly incidence and comulative number

intervention_effcs = [0.2,0.4,0.6,1.0]
interv_legend_label = [r'$20\%$ intervention efficiency',r'$40\%$ intervention efficiency',r'$60\%$ intervention efficiency',r'No intervention, schools $100\%$ occupation']
interv_color_label = ['tab:red','tab:purple','tab:orange','k']

states_ = ['S', 'E', 'I1', 'I2', 'I3', 'D', 'R']

fig, ax = plt.subplots(1,2,figsize=(14, 4))
for i, inter_ in tqdm(enumerate(intervention_effcs),total=len(intervention_effcs)):
    # read results
    if inter_ < 1.0:
        res_read = load_results_ints('soln_cum',args.population,inter_,args.school_occupation)
        res_mean = res_read.groupby('tvec').mean(); res_mean = res_mean.reset_index()
        res_loCI = res_read.groupby('tvec').quantile(0.05); res_loCI = res_loCI.reset_index()
        res_upCI = res_read.groupby('tvec').quantile(0.95); res_upCI = res_upCI.reset_index()
        res_tvec = list(res_mean['tvec'])
        res_inc  = model.get_daily_iter(res_read,res_tvec)
        res_mean_inc = res_inc.groupby('tvec').mean(); res_mean_inc = res_mean_inc.reset_index()
        res_loCI_inc = res_inc.groupby('tvec').quantile(0.05); res_loCI_inc = res_loCI_inc.reset_index()
        res_upCI_inc = res_inc.groupby('tvec').quantile(0.95); res_upCI_inc = res_upCI_inc.reset_index()
    elif inter_ == 1.0:
        res_read = load_results_dyn('soln_cum',os.path.join('results','no_intervention',str(pop),))
        res_mean = res_read.groupby('tvec').mean(); res_mean = res_mean.reset_index()
        res_loCI = res_read.groupby('tvec').quantile(0.05); res_loCI = res_loCI.reset_index()
        res_upCI = res_read.groupby('tvec').quantile(0.95); res_upCI = res_upCI.reset_index()
        res_tvec = list(res_mean['tvec'])
        res_inc  = model.get_daily_iter(res_read,res_tvec)
        res_mean_inc = res_inc.groupby('tvec').mean(); res_mean_inc = res_mean_inc.reset_index()
        res_loCI_inc = res_inc.groupby('tvec').quantile(0.05); res_loCI_inc = res_loCI_inc.reset_index()
        res_upCI_inc = res_inc.groupby('tvec').quantile(0.95); res_upCI_inc = res_upCI_inc.reset_index()

    plt.subplot(121)
    plt.plot(res_mean_inc['tvec'],res_mean_inc['E']*pop,color=interv_color_label[i],alpha=0.6)
    plt.legend(interv_legend_label,frameon=False,framealpha=0.0,bbox_to_anchor=(1,1), loc="best")
    plt.gca().set_prop_cycle(None)
    plt.fill_between(res_mean_inc['tvec'],res_loCI_inc['E']*pop,res_upCI_inc['E']*pop,color=interv_color_label[i],alpha=0.3)
    #plt.plot([20,20],[0,100],'k--',alpha=0.2)
    plt.xlim([0,max(res_mean_inc['tvec'])])
    plt.ylim([pop/pop,0.01*pop])
    plt.xticks(size=12)
    plt.yticks(size=12)
    plt.xlabel("Time (days)",size=12)
    plt.ylabel(r"Daily incidence per 100,000 ind",size=12)
    if args.type_sim == 'intervention':
        plt.title(r'Daily incidence with schools opening ${:.2f}\%$ occupation'.format(args.school_occupation*100))
    elif args.type_sim == 'school_alternancy':
        plt.title(r'Daily incidence with schools alterning ${:.2f}\%$ occupation'.format(args.school_occupation*100))
    plt.semilogy()
    plt.tight_layout()

    plt.subplot(122)
    plt.plot(res_mean['tvec'],res_mean['E']*pop,color=interv_color_label[i],alpha=0.6)
    plt.gca().set_prop_cycle(None)
    plt.fill_between(res_mean['tvec'],res_loCI['E']*pop,res_upCI['E']*pop,color=interv_color_label[i],alpha=0.3)
    plt.xlim([0,max(res_mean['tvec'])])
    plt.ylim([0,0.5*pop])
    plt.xticks(size=12)
    plt.yticks(size=12)
    plt.xlabel("Time (days)",size=12)
    plt.ylabel(r"Comulative cases per 100,000 ind",size=12)
    if args.type_sim == 'intervention':
        plt.title(r'Comulative cases with schools opening ${:.2f}\%$ occupation'.format(args.school_occupation*100))
    elif args.type_sim == 'school_alternancy':
        plt.title(r'Comulative cases with schools alterning ${:.2f}\%$ occupation'.format(args.school_occupation*100))
    # plt.semilogy()
    plt.tight_layout()

if not os.path.isdir( os.path.join(figures_path,args.type_sim) ):
    os.makedirs( os.path.join(figures_path,args.type_sim) )

save_path = os.path.join(figures_path,args.type_sim)

plt.savefig(os.path.join(figures_path,'{}_dailyIncidence_ComNumber_schoolcap_{}_n_{}.png'.format(args.type_sim,args.school_occupation,str(pop))),
            dpi=400, transparent=True, bbox_inches='tight', pad_inches=0.1 )


plt.show()



### Pointplots

intervention_effcs = [0.2,0.4,0.6]
interv_legend_label = [r'$20\%$ intervention efficiency',r'$40\%$ intervention efficiency',r'$60\%$ intervention efficiency']
school_caps        = [0.15,0.25,0.35,0.55,1.0]
df_list = []
for i, inter_ in enumerate(intervention_effcs):
    for j, cap_ in enumerate(school_caps):
        res_read = load_results_ints('soln',args.population,inter_,cap_)
        res_smooth = model.smooth_timecourse(res_read)
        for itr_ in range(10):
            res_smooth_i = res_smooth['iter'] == itr_
            res_smooth_i = pd.DataFrame(res_smooth[res_smooth_i])
            peak_I3 = max(res_smooth_i['I3'])*pop

            df_res_i = pd.DataFrame(columns=['iter','interven_eff','school_cap','peak_I3'])
            df_res_i['iter']         = [int(itr_)]
            df_res_i['interven_eff'] = r'{}$\%$'.format(int(inter_*100))
            df_res_i['school_cap']   = int(cap_*100)
            df_res_i['peak_I3']      = peak_I3
            df_list.append(df_res_i)

df_peaks_I3 = pd.concat(df_list)


plt.figure(figsize=(6, 5))
sns.pointplot( data=df_peaks_I3, x='school_cap', y='peak_I3', hue='interven_eff', linestyles='')
#plt.legend(frameon=False,framealpha=0.0,bbox_to_anchor=(0.5,-0.2), loc="lower center")
plt.xlabel(r'School capacity ($\%$)',size=12)
plt.ylabel(r'ICUs required in peak',size=12)
plt.xticks(size=12)
plt.yticks(size=12)
plt.show()


# res_read = load_results_ints('soln',args.population,intervention_effcs[2],args.school_occupation)
# res_smooth = model.smooth_timecourse(res_read)
# res_mean = res_smooth.groupby('tvec').mean(); res_mean = res_mean.reset_index()

# peak UCIs



#res_mean = res_read.groupby('tvec').mean(); res_mean = res_mean.reset_index()

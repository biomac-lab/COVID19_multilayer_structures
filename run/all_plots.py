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

parser.add_argument('--population', default=10000, type=int,
                    help='Speficy the number of individials')
parser.add_argument('--type_sim', default='intervention', type=str,
                    help='Speficy the type of simulation to plot')
args = parser.parse_args()

number_nodes = args.population
pop = number_nodes

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
    read_path = os.path.join(path,'{}_inter_{}_schoolcap_{}_mask_N95_peopleMasked_1.0_ventilation_3_ID_ND_{}.csv'.format(str(n),str(int_effec),
                                                                           str(schl_occup),type_res))
    read_file = pd.read_csv(read_path)
    return read_file

########################################################################################################################################
########################################################################################################################################
########################################################################################################################################
########################################################################################################################################
########################################################################################################################################

### Plot new cases

results_path = os.path.join(results_path,'intervention',str(pop))

intervention_effcs = [0.0,0.2,0.4,0.6] #,1.0]
interv_legend_label = [r'$0\%$ intervention efficiency',r'$20\%$ intervention efficiency',r'$40\%$ intervention efficiency',r'$60\%$ intervention efficiency'] #,r'No intervention, schools $100\%$ occupation']
interv_color_label = ['k','tab:red','tab:purple','tab:orange']

school_caps        = [0.35] #[0.15,0.25,0.35,0.55,1.0]

states_ = ['S', 'E', 'I1', 'I2', 'I3', 'D', 'R']
plot_state = 'E'

alpha = 0.05
# lineal
for c, cap_ in tqdm(enumerate(school_caps), total=len(school_caps)):
    plt.figure(figsize=(6,4))  # create figure
    for i, inter_ in enumerate(intervention_effcs):
        # read results
        # if inter_ < 1.0:
        res_read = load_results_ints('soln',args.population,inter_,cap_,path=results_path)
        res_median = res_read.groupby('tvec').median(); res_median = res_median.reset_index()
        res_loCI = res_read.groupby('tvec').quantile(alpha/2); res_loCI = res_loCI.reset_index()
        res_upCI = res_read.groupby('tvec').quantile(1-alpha/2); res_upCI = res_upCI.reset_index()
        # read results with no intervention
        # elif inter_ == 1.0:
        #     res_read = load_results_dyn('soln',os.path.join('results','no_intervention',str(pop)))
        #     res_median = res_read.groupby('tvec').median(); res_median = res_median.reset_index()
        #     res_loCI = res_read.groupby('tvec').quantile(alpha/2); res_loCI = res_loCI.reset_index()
        #     res_upCI = res_read.groupby('tvec').quantile(1-alpha/2); res_upCI = res_upCI.reset_index()
        # plot
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
            plt.title(r'New cases with schools opening ${}\%$ occupation'.format(int(cap_*100)))
        elif args.type_sim == 'school_alternancy':
            plt.title(r'New cases with schools alterning ${}\%$ occupation'.format(int(cap_*100)))

    if not os.path.isdir( os.path.join(figures_path,'cases_evolution') ):
        os.makedirs( os.path.join(figures_path,'cases_evolution') )

    save_path = os.path.join(figures_path,'cases_evolution','{}_lin_{}_dynamics_schoolcap_{}_n_{}.png'.format(plot_state,args.type_sim,cap_,str(pop)))

    #plt.savefig(save_path,dpi=400, transparent=True, bbox_inches='tight', pad_inches=0.1 )
    plt.show()

school_caps        = [0.35] #[0.15,0.25,0.35,0.55,1.0]
plot_state = 'D'
alpha = 0.05
# lineal
for c, cap_ in tqdm(enumerate(school_caps), total=len(school_caps)):
    plt.figure(figsize=(6,4))  # create figure
    for i, inter_ in enumerate(intervention_effcs):
        # read results
        if inter_ < 1.0:
            res_read = load_results_ints('soln',args.population,inter_,cap_,path=results_path)
            res_median = res_read.groupby('tvec').median(); res_median = res_median.reset_index()
            res_loCI = res_read.groupby('tvec').quantile(alpha/2); res_loCI = res_loCI.reset_index()
            res_upCI = res_read.groupby('tvec').quantile(1-alpha/2); res_upCI = res_upCI.reset_index()
        # read results with no intervention
        elif inter_ == 1.0:
            res_read = load_results_dyn('soln',os.path.join('results','no_intervention',str(pop)))
            res_median = res_read.groupby('tvec').median(); res_median = res_median.reset_index()
            res_loCI = res_read.groupby('tvec').quantile(alpha/2); res_loCI = res_loCI.reset_index()
            res_upCI = res_read.groupby('tvec').quantile(1-alpha/2); res_upCI = res_upCI.reset_index()
        # plot
        plt.plot(res_median['tvec'],res_median[plot_state]*pop,color=interv_color_label[i])
        plt.legend(interv_legend_label,frameon=False,framealpha=0.0,bbox_to_anchor=(1,1), loc="best")
        plt.gca().set_prop_cycle(None)
        plt.fill_between(res_median['tvec'],res_loCI[plot_state]*pop,res_upCI[plot_state]*pop,color=interv_color_label[i],alpha=0.3)
        plt.axvspan(0,20,color='gray',alpha=0.05)
        plt.annotate('Schools \n closed',(0,150),size=9)
        plt.annotate('Schools \n open',(22,150),size=9)
        plt.xlim([0,max(res_median['tvec'])])
        plt.ylim([0,0.02*pop])
        plt.xticks(size=12)
        plt.yticks(size=12)
        plt.xlabel("Time (days)",size=12)
        plt.ylabel(r"Deaths per 10,000 ind",size=12)
        if args.type_sim == 'intervention':
            plt.title(r'Deaths with schools opening ${}\%$ occupation'.format(int(cap_*100)))
        elif args.type_sim == 'school_alternancy':
            plt.title(r'Deaths with schools alterning ${}\%$ occupation'.format(int(cap_*100)))

    if not os.path.isdir( os.path.join(figures_path,'cases_evolution') ):
        os.makedirs( os.path.join(figures_path,'cases_evolution') )

    save_path = os.path.join(figures_path,'cases_evolution','{}_lin_{}_dynamics_schoolcap_{}_n_{}.png'.format(plot_state,args.type_sim,cap_,str(pop)))

    plt.savefig(save_path,dpi=400, transparent=True, bbox_inches='tight', pad_inches=0.1 )
    #plt.show()

# logaritmic
plot_state = 'E'
for c, cap_ in tqdm(enumerate(school_caps), total=len(school_caps)):
    plt.figure(figsize=(6,4))  # create figure
    for i, inter_ in enumerate(intervention_effcs):
        # read results
        if inter_ < 1.0:
            res_read = load_results_ints('soln',args.population,inter_,cap_,path=results_path)
            res_median = res_read.groupby('tvec').median(); res_median = res_median.reset_index()
            res_loCI = res_read.groupby('tvec').quantile(alpha/2); res_loCI = res_loCI.reset_index()
            res_upCI = res_read.groupby('tvec').quantile(1-alpha/2); res_upCI = res_upCI.reset_index()
        # read results with no intervention
        elif inter_ == 1.0:
            res_read = load_results_dyn('soln',os.path.join('results','no_intervention',str(pop)))
            res_median = res_read.groupby('tvec').median(); res_median = res_median.reset_index()
            res_loCI = res_read.groupby('tvec').quantile(alpha/2); res_loCI = res_loCI.reset_index()
            res_upCI = res_read.groupby('tvec').quantile(1-alpha/2); res_upCI = res_upCI.reset_index()
        # plot
        plt.plot(res_median['tvec'],res_median['E']*100,color=interv_color_label[i])
        plt.legend(interv_legend_label,frameon=False,framealpha=0.0,bbox_to_anchor=(1,1), loc="best")
        plt.gca().set_prop_cycle(None)
        plt.fill_between(res_median['tvec'],res_loCI['E']*100,res_upCI['E']*100,color=interv_color_label[i],alpha=0.3)
        plt.axvspan(0,20,color='gray',alpha=0.05)
        plt.annotate('Schools \n closed',(0,1.2),size=9)
        plt.annotate('Schools \n open',(22,1.2),size=9)
        plt.xlim([0,max(res_median['tvec'])])
        plt.ylim([1/pop*200,1*100])
        plt.xticks(size=12)
        plt.yticks(size=12)
        plt.xlabel("Time (days)",size=12)
        plt.ylabel(r"$\%$ new cases per 100,000 ind",size=12)
        plt.semilogy()
        plt.tight_layout()
        if args.type_sim == 'intervention':
            plt.title(r'New cases with schools opening ${}\%$ occupation'.format(int(cap_*100)))
        elif args.type_sim == 'school_alternancy':
            plt.title(r'New cases with schools alterning ${}\%$ occupation'.format(int(cap_*100)))

    if not os.path.isdir( os.path.join(figures_path,'cases_evolution') ):
        os.makedirs( os.path.join(figures_path,'cases_evolution') )

    save_path = os.path.join(figures_path,'cases_evolution','{}_log_{}_dynamics_schoolcap_{}_n_{}.png'.format(plot_state,args.type_sim,cap_,str(pop)))

    plt.savefig(save_path,dpi=400, transparent=True, bbox_inches='tight', pad_inches=0.1 )
    #plt.show()

# # logaritmic
# plot_state = 'D'
# for c, cap_ in tqdm(enumerate(school_caps), total=len(school_caps)):
#     plt.figure(figsize=(6,4))  # create figure
#     for i, inter_ in enumerate(intervention_effcs):
#         # read results
#         if inter_ < 1.0:
#             res_read = load_results_ints('soln',args.population,inter_,cap_,path=results_path)
#             res_median = res_read.groupby('tvec').median(); res_median = res_median.reset_index()
#             res_loCI = res_read.groupby('tvec').quantile(alpha/2); res_loCI = res_loCI.reset_index()
#             res_upCI = res_read.groupby('tvec').quantile(1-alpha/2); res_upCI = res_upCI.reset_index()
#         # read results with no intervention
#         elif inter_ == 1.0:
#             res_read = load_results_dyn('soln',os.path.join('results','no_intervention',str(pop)))
#             res_median = res_read.groupby('tvec').median(); res_median = res_median.reset_index()
#             res_loCI = res_read.groupby('tvec').quantile(alpha/2); res_loCI = res_loCI.reset_index()
#             res_upCI = res_read.groupby('tvec').quantile(1-alpha/2); res_upCI = res_upCI.reset_index()
#         # plot
#         plt.plot(res_median['tvec'],res_median[plot_state]*100,color=interv_color_label[i])
#         plt.legend(interv_legend_label,frameon=False,framealpha=0.0,bbox_to_anchor=(1,1), loc="best")
#         plt.gca().set_prop_cycle(None)
#         plt.fill_between(res_median['tvec'],res_loCI[plot_state]*100,res_upCI[plot_state]*100,color=interv_color_label[i],alpha=0.3)
#         plt.axvspan(0,20,color='gray',alpha=0.05)
#         plt.annotate('Schools \n closed',(0,1.2),size=9)
#         plt.annotate('Schools \n open',(22,1.2),size=9)
#         plt.xlim([0,max(res_median['tvec'])])
#         plt.ylim([1/pop*200,2*100])
#         plt.xticks(size=12)
#         plt.yticks(size=12)
#         plt.xlabel("Time (days)",size=12)
#         plt.ylabel(r"$\%$ Deaths per 100,000 ind",size=12)
#         plt.semilogy()
#         plt.tight_layout()
#         if args.type_sim == 'intervention':
#             plt.title(r'Deaths with schools opening ${}\%$ occupation'.format(int(cap_*100)))
#         elif args.type_sim == 'school_alternancy':
#             plt.title(r'Deaths with schools alterning ${}\%$ occupation'.format(int(cap_*100)))

#     if not os.path.isdir( os.path.join(figures_path,'cases_evolution') ):
#         os.makedirs( os.path.join(figures_path,'cases_evolution') )

#     save_path = os.path.join(figures_path,'cases_evolution','{}_log_{}_dynamics_schoolcap_{}_n_{}.png'.format(plot_state,args.type_sim,cap_,str(pop)))

#     #plt.savefig(save_path,dpi=400, transparent=True, bbox_inches='tight', pad_inches=0.1 )
# plt.show()


########################################################################################################################################
########################################################################################################################################
########################################################################################################################################
########################################################################################################################################
########################################################################################################################################

### Plot dayly incidence

intervention_effcs = [0.2,0.4,0.6,1.0]
interv_legend_label = [r'$20\%$ intervention efficiency',r'$40\%$ intervention efficiency',r'$60\%$ intervention efficiency',r'No intervention, schools $100\%$ occupation']
interv_color_label = ['tab:red','tab:purple','tab:orange','k']

states_ = ['S', 'E', 'I1', 'I2', 'I3', 'D', 'R']
plot_state = 'E'

school_caps        = [0.35]#[0.15,0.25,0.35,0.55,1.0]

alpha = 0.05

# lineal

for c, cap_ in tqdm(enumerate(school_caps), total=len(school_caps)):
    plt.figure(figsize=(6,4))  # create figure
    for i, inter_ in enumerate(intervention_effcs):
        # read results
        if inter_ < 1.0:
            res_read = load_results_ints('soln_cum',args.population,inter_,cap_,results_path)
            res_median = res_read.groupby('tvec').median(); res_median = res_median.reset_index()
            res_loCI = res_read.groupby('tvec').quantile(alpha/2); res_loCI = res_loCI.reset_index()
            res_upCI = res_read.groupby('tvec').quantile(1-alpha/2); res_upCI = res_upCI.reset_index()
            res_tvec = list(res_median['tvec'])
            res_inc  = model.get_daily_iter(res_read,res_tvec)
            res_median_inc = res_inc.groupby('tvec').median(); res_median_inc = res_median_inc.reset_index()
            res_loCI_inc = res_inc.groupby('tvec').quantile(alpha/2); res_loCI_inc = res_loCI_inc.reset_index()
            res_upCI_inc = res_inc.groupby('tvec').quantile(1-alpha/2); res_upCI_inc = res_upCI_inc.reset_index()
        # read results with no intervention
        elif inter_ == 1.0:
            res_read = load_results_dyn('soln',os.path.join('results','no_intervention',str(pop)))
            res_median = res_read.groupby('tvec').median(); res_median = res_median.reset_index()
            res_loCI = res_read.groupby('tvec').quantile(alpha/2); res_loCI = res_loCI.reset_index()
            res_upCI = res_read.groupby('tvec').quantile(1-alpha/2); res_upCI = res_upCI.reset_index()
            res_tvec = list(res_median['tvec'])
            res_inc  = model.get_daily_iter(res_read,res_tvec)
            res_median_inc = res_inc.groupby('tvec').median(); res_median_inc = res_median_inc.reset_index()
            res_loCI_inc = res_inc.groupby('tvec').quantile(alpha/2); res_loCI_inc = res_loCI_inc.reset_index()
            res_upCI_inc = res_inc.groupby('tvec').quantile(1-alpha/2); res_upCI_inc = res_upCI_inc.reset_index()
        # plot
        plt.plot(res_median_inc['tvec'],res_median_inc[plot_state]*pop,color=interv_color_label[i],alpha=0.6)
        plt.legend(interv_legend_label,frameon=False,framealpha=0.0,bbox_to_anchor=(1,1), loc="best")
        plt.gca().set_prop_cycle(None)
        plt.fill_between(res_median_inc['tvec'],res_loCI_inc[plot_state]*pop,res_upCI_inc[plot_state]*pop,color=interv_color_label[i],alpha=0.3)
        plt.axvspan(0,20,color='k',alpha=0.035)
        plt.annotate('Schools \n closed',(0,8))
        plt.xlim([0,max(res_median_inc['tvec'])])
        plt.ylim([0,0.2*pop])
        plt.xticks(size=12)
        plt.yticks(size=12)
        plt.xlabel("Time (days)",size=12)
        plt.ylabel(r"Daily incidence per 100,000 ind",size=12)
        if args.type_sim == 'intervention':
            plt.title(r'Daily incidence with schools opening ${:.2f}\%$ occupation'.format(cap_*100))
        elif args.type_sim == 'school_alternancy':
            plt.title(r'Daily incidence with schools alterning ${:.2f}\%$ occupation'.format(cap_*100))
        plt.semilogy()
        plt.tight_layout()

    if not os.path.isdir( os.path.join(figures_path,'daily_incidence') ):
        os.makedirs( os.path.join(figures_path,'daily_incidence') )

    save_path = os.path.join(figures_path,'daily_incidence','{}_lin_{}_dynamics_schoolcap_{}_n_{}.png'.format(plot_state,args.type_sim,cap_,str(pop)))

    #plt.savefig(save_path,dpi=400, transparent=True, bbox_inches='tight', pad_inches=0.1 )
plt.show()

# log

for c, cap_ in tqdm(enumerate(school_caps), total=len(school_caps)):
    plt.figure(figsize=(6,4))  # create figure
    for i, inter_ in enumerate(intervention_effcs):
        # read results
        if inter_ < 1.0:
            res_read = load_results_ints('soln_cum',args.population,inter_,cap_,results_path)
            res_median = res_read.groupby('tvec').median(); res_median = res_median.reset_index()
            res_loCI = res_read.groupby('tvec').quantile(alpha/2); res_loCI = res_loCI.reset_index()
            res_upCI = res_read.groupby('tvec').quantile(1-alpha/2); res_upCI = res_upCI.reset_index()
            res_tvec = list(res_median['tvec'])
            res_inc  = model.get_daily_iter(res_read,res_tvec)
            res_median_inc = res_inc.groupby('tvec').median(); res_median_inc = res_median_inc.reset_index()
            res_loCI_inc = res_inc.groupby('tvec').quantile(alpha/2); res_loCI_inc = res_loCI_inc.reset_index()
            res_upCI_inc = res_inc.groupby('tvec').quantile(1-alpha/2); res_upCI_inc = res_upCI_inc.reset_index()
        # read results with no intervention
        elif inter_ == 1.0:
            res_read = load_results_dyn('soln_cum',os.path.join('results','no_intervention',str(pop)))
            res_median = res_read.groupby('tvec').median(); res_median = res_median.reset_index()
            res_loCI = res_read.groupby('tvec').quantile(alpha/2); res_loCI = res_loCI.reset_index()
            res_upCI = res_read.groupby('tvec').quantile(1-alpha/2); res_upCI = res_upCI.reset_index()
            res_tvec = list(res_median['tvec'])
            res_inc  = model.get_daily_iter(res_read,res_tvec)
            res_median_inc = res_inc.groupby('tvec').median(); res_median_inc = res_median_inc.reset_index()
            res_loCI_inc = res_inc.groupby('tvec').quantile(alpha/2); res_loCI_inc = res_loCI_inc.reset_index()
            res_upCI_inc = res_inc.groupby('tvec').quantile(1-alpha/2); res_upCI_inc = res_upCI_inc.reset_index()
        # plot
        plt.plot(res_median_inc['tvec'],res_median_inc[plot_state]*pop,color=interv_color_label[i],alpha=0.6)
        plt.legend(interv_legend_label,frameon=False,framealpha=0.0,bbox_to_anchor=(1,1), loc="best")
        plt.gca().set_prop_cycle(None)
        plt.fill_between(res_median_inc['tvec'],res_loCI_inc[plot_state]*pop,res_upCI_inc[plot_state]*pop,color=interv_color_label[i],alpha=0.3)
        plt.axvspan(0,20,color='k',alpha=0.035)
        plt.annotate('Schools \n closed',(0,8))
        plt.xlim([0,max(res_median_inc['tvec'])])
        plt.ylim([pop/pop,0.01*pop])
        plt.xticks(size=12)
        plt.yticks(size=12)
        plt.xlabel("Time (days)",size=12)
        plt.ylabel(r"Daily incidence per 100,000 ind",size=12)
        if args.type_sim == 'intervention':
            plt.title(r'Daily incidence with schools opening ${:.2f}\%$ occupation'.format(cap_*100))
        elif args.type_sim == 'school_alternancy':
            plt.title(r'Daily incidence with schools alterning ${:.2f}\%$ occupation'.format(cap_*100))
        plt.semilogy()
        plt.tight_layout()

    if not os.path.isdir( os.path.join(figures_path,'daily_incidence') ):
        os.makedirs( os.path.join(figures_path,'daily_incidence') )

    save_path = os.path.join(figures_path,'daily_incidence','{}_log_{}_dynamics_schoolcap_{}_n_{}.png'.format(plot_state,args.type_sim,cap_,str(pop)))

    #plt.savefig(save_path,dpi=400, transparent=True, bbox_inches='tight', pad_inches=0.1 )
plt.show()


########################################################################################################################################
########################################################################################################################################
########################################################################################################################################
########################################################################################################################################
########################################################################################################################################

### PLot comulative number



intervention_effcs = [0.0,0.2,0.4,0.6] #,1.0]
interv_legend_label = [r'$0\%$ intervention efficiency',r'$20\%$ intervention efficiency',r'$40\%$ intervention efficiency',r'$60\%$ intervention efficiency'] #,r'No intervention, schools $100\%$ occupation']
interv_color_label = ['k','tab:red','tab:purple','tab:orange']

school_caps        = [0.35] #[0.15,0.25,0.35,0.55,1.0]

states_ = ['S', 'E', 'I1', 'I2', 'I3', 'D', 'R']
plot_state = 'E'
plt.figure(figsize=(6,4))  # create figure
for c, cap_ in tqdm(enumerate(school_caps), total=len(school_caps)):
    for i, inter_ in enumerate(intervention_effcs):
        # read results
        if inter_ < 1.0:
            res_read = load_results_ints('soln_cum',args.population,inter_,cap_,results_path)
            res_median = res_read.groupby('tvec').median(); res_median = res_median.reset_index()
            res_loCI = res_read.groupby('tvec').quantile(alpha/2); res_loCI = res_loCI.reset_index()
            res_upCI = res_read.groupby('tvec').quantile(1-alpha/2); res_upCI = res_upCI.reset_index()
        # read results with no intervention
        elif inter_ == 1.0:
            res_read = load_results_dyn('soln_cum',os.path.join('results','no_intervention',str(pop)))
            res_median = res_read.groupby('tvec').median(); res_median = res_median.reset_index()
            res_loCI = res_read.groupby('tvec').quantile(alpha/2); res_loCI = res_loCI.reset_index()
            res_upCI = res_read.groupby('tvec').quantile(1-alpha/2); res_upCI = res_upCI.reset_index()
            res_tvec = list(res_median['tvec'])
        # plot
        plt.plot(res_median['tvec'],res_median[plot_state]*pop,color=interv_color_label[i],alpha=0.6)
        plt.legend(interv_legend_label,frameon=False,framealpha=0.0,bbox_to_anchor=(1,1), loc="best")
        plt.gca().set_prop_cycle(None)
        plt.fill_between(res_median['tvec'],res_loCI[plot_state]*pop,res_upCI[plot_state]*pop,color=interv_color_label[i],alpha=0.3)
        plt.axvspan(0,20,color='k',alpha=0.035)
        plt.annotate('Schools \n closed',(0,6000),size=9)
        plt.annotate('Schools \n open',(22,6000),size=9)
        plt.xlim([0,max(res_median['tvec'])])
        plt.ylim([0,0.9*pop])
        plt.xticks(size=12)
        plt.yticks(size=12)
        plt.xlabel("Time (days)",size=12)
        plt.ylabel(r"Cumulative cases per 10,000 ind",size=12)
        if args.type_sim == 'intervention':
            plt.title(r'Cumulative cases with schools opening ${}\%$ occupation'.format(int(cap_*100)))
        elif args.type_sim == 'school_alternancy':
            plt.title(r'Cumulative cases with schools alterning ${}\%$ occupation'.format(int(cap_*100)))
        plt.tight_layout()

        if not os.path.isdir( os.path.join(figures_path,'comulative_cases') ):
            os.makedirs( os.path.join(figures_path,'comulative_cases') )

    save_path = os.path.join(figures_path,'comulative_cases','{}_lin_{}_dynamics_schoolcap_{}_n_{}.png'.format(plot_state,args.type_sim,cap_,str(pop)))

    plt.savefig(save_path,dpi=400, transparent=True, bbox_inches='tight', pad_inches=0.1 )

#plt.show()


states_ = ['S', 'E', 'I1', 'I2', 'I3', 'D', 'R']
plot_state = 'D'
plt.figure(figsize=(6,4))  # create figure
for c, cap_ in tqdm(enumerate(school_caps), total=len(school_caps)):
    for i, inter_ in enumerate(intervention_effcs):
        # read results
        if inter_ < 1.0:
            res_read = load_results_ints('soln_cum',args.population,inter_,cap_,results_path)
            res_median = res_read.groupby('tvec').median(); res_median = res_median.reset_index()
            res_loCI = res_read.groupby('tvec').quantile(alpha/2); res_loCI = res_loCI.reset_index()
            res_upCI = res_read.groupby('tvec').quantile(1-alpha/2); res_upCI = res_upCI.reset_index()
        # read results with no intervention
        elif inter_ == 1.0:
            res_read = load_results_dyn('soln_cum',os.path.join('results','no_intervention',str(pop)))
            res_median = res_read.groupby('tvec').median(); res_median = res_median.reset_index()
            res_loCI = res_read.groupby('tvec').quantile(alpha/2); res_loCI = res_loCI.reset_index()
            res_upCI = res_read.groupby('tvec').quantile(1-alpha/2); res_upCI = res_upCI.reset_index()
            res_tvec = list(res_median['tvec'])
        # plot
        plt.plot(res_median['tvec'],res_median[plot_state]*pop,color=interv_color_label[i],alpha=0.6)
        plt.legend(interv_legend_label,frameon=False,framealpha=0.0,bbox_to_anchor=(1,1), loc="best")
        plt.gca().set_prop_cycle(None)
        plt.fill_between(res_median['tvec'],res_loCI[plot_state]*pop,res_upCI[plot_state]*pop,color=interv_color_label[i],alpha=0.3)
        plt.axvspan(0,20,color='gray',alpha=0.05)
        plt.annotate('Schools \n closed',(0,20000),size=9)
        plt.annotate('Schools \n open',(22,20000),size=9)
        plt.xlim([0,max(res_median['tvec'])])
        plt.ylim([0,0.02*pop])
        plt.xticks(size=12)
        plt.yticks(size=12)
        plt.xlabel("Time (days)",size=12)
        plt.ylabel(r"Comulative deaths per 100,000 ind",size=12)
        if args.type_sim == 'intervention':
            plt.title(r'Comulative deaths with schools opening ${}\%$ occupation'.format(int(cap_*100)))
        elif args.type_sim == 'school_alternancy':
            plt.title(r'Comulative deaths with schools alterning ${}\%$ occupation'.format(int(cap_*100)))

    if not os.path.isdir( os.path.join(figures_path,'comulative_cases') ):
        os.makedirs( os.path.join(figures_path,'comulative_cases') )

    save_path = os.path.join(figures_path,'comulative_cases','{}_lin_{}_dynamics_schoolcap_{}_n_{}.png'.format(plot_state,args.type_sim,cap_,str(pop)))

    plt.savefig(save_path,dpi=400, transparent=True, bbox_inches='tight', pad_inches=0.1 )

plt.show()
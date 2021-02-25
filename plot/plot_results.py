import pandas as pd
import matplotlib.pyplot as plt
import os

config_data = pd.read_csv('configlin.csv', sep=',', header=None, index_col=0)
figures_path = config_data.loc['figures_dir'][1]
results_path = config_data.loc['results_dir'][1]
ages_data_path = config_data.loc['bogota_age_data_dir'][1]
houses_data_path = config_data.loc['bogota_houses_data_dir'][1]


import argparse

parser = argparse.ArgumentParser(description='Dynamics visualization.')

parser.add_argument('--population', default=100000, type=int,
                    help='Speficy the number of individials')
parser.add_argument('--type_sim', default='no_intervention', type=str,
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

alpha = 0.05
res_read = load_results_dyn('soln')
res_median = res_read.groupby('tvec').median(); res_median = res_median.reset_index()
res_loCI = res_read.groupby('tvec').quantile(alpha/2); res_loCI = res_loCI.reset_index()
res_upCI = res_read.groupby('tvec').quantile(1-alpha/2); res_upCI = res_upCI.reset_index()

def plot_state_dynamics(soln_avg=res_median,soln_loCI=res_loCI,soln_upCI=res_upCI,scale=1,ymax=1,n=args.population,saveFig=False):

    tvec = res_median['tvec']
    states_ = ['S', 'E', 'I1', 'I2', 'I3', 'D', 'R']

    # plot linear
    plt.figure(figsize=(2*6.4, 4.0))
    plt.subplot(121)
    plt.plot(tvec,soln_avg[states_]*scale)
    plt.legend(states_,frameon=False,framealpha=0.0,bbox_to_anchor=(1.04,1), loc="upper left")
    # add ranges
    plt.gca().set_prop_cycle(None)
    for i, s in enumerate(states_):
        plt.fill_between(tvec,soln_loCI[str(s)]*scale,soln_upCI[str(s)]*scale,alpha=0.3)

    plt.ylim([0,ymax*scale])
    plt.ylim([0,0.1])
    plt.xlabel("Time (days)")
    plt.ylabel("Number")

    # plot log
    plt.subplot(122)
    plt.plot(tvec,soln_avg[states_]*scale)
    plt.legend(states_,frameon=False,framealpha=0.0,bbox_to_anchor=(1.04,1), loc="upper left")
    # add ranges
    plt.gca().set_prop_cycle(None)
    for i, s in enumerate(states_):
        plt.fill_between(tvec,soln_loCI[str(s)]*scale,soln_upCI[str(s)]*scale,alpha=0.3)

    plt.ylim([scale/n,ymax*scale])
    plt.xlabel("Time (days)")
    plt.ylabel("Number")
    plt.semilogy()
    plt.tight_layout()

    if saveFig == False:
        plt.show()
    else:
        if not os.path.isdir( os.path.join(figures_path, str(number_nodes)) ):
            os.makedirs(os.path.join(figures_path, str(number_nodes)))
        
        plt.savefig(os.path.join(figures_path, 'dynamics_n_{}.png'.format(number_nodes)), dpi=400, transparent=False, bbox_inches = 'tight', pad_inches = 0.1)


# Plot
plot_state_dynamics()
import pandas as pd
import matplotlib.pyplot as plt
import os

config_data = pd.read_csv('configlin.csv', sep=',', header=None, index_col=0)
figures_path = config_data.loc['figures_dir'][1]
results_path = config_data.loc['results_dir'][1]
ages_data_path = config_data.loc['bogota_age_data_dir'][1]
houses_data_path = config_data.loc['bogota_houses_data_dir'][1]


import argparse

parser = argparse.ArgumentParser(description='Networks visualization.')

parser.add_argument('--population', default=1000, type=int,
                    help='Speficy the number of individials')

args = parser.parse_args()

number_nodes = args.population
pop = number_nodes

def load_results(type,path=results_path,n=pop):
    read_path = os.path.join(path,str(n),'{}_{}.csv'.format(str(n),str(type)))
    read_file = pd.read_csv(read_path)
    return read_file

mean_res = load_results(type='mean')
loCI     = load_results(type='loCI')
upCI     = load_results(type='upCI')


def plot_state_dynamics(soln_avg=mean_res,soln_loCI=loCI,soln_upCI=upCI,scale=1,ymax=1,n=args.population,saveFig=False):

    tvec = mean_res['tvec']
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
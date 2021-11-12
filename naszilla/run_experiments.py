import argparse
import time
import logging
import sys
import os
import pickle
import numpy as np
import copy
import warnings
warnings.filterwarnings("ignore")


from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator

sys.path.insert(0, './')
sys.path.insert(0, '../')

from naszilla.params import *
from naszilla.nas_benchmarks import Nasbench101

# No support for Nasbench201 and Nasbench301 for now
# from naszilla.nas_benchmarks import Nasbench201, Nasbench301

from naszilla.nas_algorithms import run_nas_algorithm

algo_labels = {'random': 'RS', 'evolution': 'ES', 'bananas': 'BANANAS', 'bonas': 'BONAS', 'gp_bayesopt': 'GP-BO',
    'dngo': 'DNGO', 'bohamiann': 'BOHAMIANN', 'local_search': 'LS', 'gcn_predictor': 'GCN', 'boshnas': 'BOSHNAS'}

def run_experiments(args, save_dir):

    # set up arguments
    trials = args.trials
    queries = args.queries
    out_file = args.output_filename
    save_specs = args.save_specs
    metann_params = meta_neuralnet_params(args.metann_params)
    ss = args.search_space
    dataset = args.dataset
    mf = args.mf
    algorithm_params = algo_params(args.algo_params, queries=queries)
    num_algos = len(algorithm_params)
    logging.info(algorithm_params)

    # set up search space
    mp = copy.deepcopy(metann_params)

    if ss == 'nasbench_101':
        search_space = Nasbench101(mf=mf)
    # No support for Nasbench201 and Nasbench301 for now
    # elif ss == 'nasbench_201':
    #     search_space = Nasbench201(dataset=dataset)
    # elif ss == 'nasbench_301':
    #     search_space = Nasbench301()
    elif ss == 'nasbench_201' or ss == 'nasbenc_301':
    	raise NotImplementedError('No support for Nasbench201 and Nasbench301 for now')
    else:
        raise ValueError('Invalid search space')

    # for i in range(trials):
    #     results = []
    #     val_results = []
    #     walltimes = []
    #     run_data = []

    #     for j in range(num_algos):
    #         print('\n* Running NAS algorithm: {}'.format(algorithm_params[j]))
    #         starttime = time.time()
    #         # this line runs the nas algorithm and returns the result
    #         result, val_result, run_datum = run_nas_algorithm(algorithm_params[j], search_space, mp)

    #         result = np.round(result, 5)
    #         val_result = np.round(val_result, 5)

    #         # remove unnecessary dict entries that take up space
    #         for d in run_datum:
    #             if not save_specs:
    #                 d.pop('spec')
    #             for key in ['encoding', 'adj', 'path', 'dist_to_min']:
    #                 if key in d:
    #                     d.pop(key)

    #         # add walltime, results, run_data
    #         walltimes.append(time.time()-starttime)
    #         results.append(result)
    #         val_results.append(val_result)
    #         run_data.append(run_datum)

    #     # print and pickle results
    #     filename = os.path.join(save_dir, '{}_{}.pkl'.format(out_file, i))
    #     print('\n* Trial summary: (params, results, walltimes)')
    #     print(algorithm_params)
    #     print(ss)
    #     print(results)
    #     print(walltimes)
    #     print('\n* Saving to file {}'.format(filename))
    #     with open(filename, 'wb') as f:
    #         pickle.dump([algorithm_params, metann_params, results, walltimes, run_data, val_results], f)
    #         f.close()

    cp = True
    eps = 0.1

    if args.save_fig:
        losses = {param['algo_name']:[] for param in algorithm_params}
        for i in range(trials):
            best_loss = 100
            boshnas_loss = 100
            filename = os.path.join(save_dir, '{}_{}.pkl'.format(out_file, i))
            with open(filename, 'rb') as f:
                results = pickle.load(f)
                for j in range(len(results[0])):
                    losses[results[0][j]['algo_name']].append(results[2][j])
                    best_loss = min(best_loss, results[2][j][-1, -1])
                    if results[0][j]['algo_name'] == 'boshnas': 
                        boshnas_loss = results[2][j][-1, -1]
            if cp:
                if boshnas_loss > best_loss + eps:
                    for algo in losses.keys():
                        losses[algo].pop()

        for algo, loss in losses.items():
        	num_trials = len(loss); break

        fig, ax = plt.subplots(figsize=(6.4, 4))
        for algo in losses.keys():
            ax.errorbar(losses[algo][0][:, 0], 
                         np.mean([loss[:, 1] for loss in losses[algo]], axis=0),
                         yerr=1.64*np.std([loss[:, 1] for loss in losses[algo]], axis=0),
                         marker='o',
                         capsize=5,
                         label=algo_labels[algo])
        ax.grid()
        ax.set_xlabel('Queries', fontsize=12)
        ax.set_ylabel('Test Loss', fontsize=12)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.legend(ncol=2 if args.algo_params == 'all_algos' else 1)
        fig_filename = os.path.join(save_dir, 'results.pdf') 
        fig.savefig(fig_filename)
        plt.close(fig)

        print('\n* Saving figure to {} using {} trials'.format(fig_filename, num_trials))


def main(args):
    # Delete pngs in main
    for file in os.listdir('./'):
        if '.png' in file: os.remove(file)

    # make save directory
    save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    algo_params = args.algo_params
    save_path = save_dir + '/' + algo_params + '/'
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    # set up logging
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(save_dir, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)
    logging.info(args)

    run_experiments(args, save_path)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Args for BANANAS experiments')
    parser.add_argument('--trials', type=int, default=500, help='Number of trials')
    parser.add_argument('--queries', type=int, default=150, help='Max number of queries/evaluations each NAS algorithm gets')
    parser.add_argument('--search_space', type=str, default='nasbench_101', help='nasbench_101, _201, or _301')
    parser.add_argument('--dataset', type=str, default='cifar10', help='cifar10, 100, or imagenet (for nasbench201)')
    parser.add_argument('--mf', type=bool, default=False, help='Multi fidelity: true or false (for nasbench101)')
    parser.add_argument('--metann_params', type=str, default='standard', help='which parameters to use')
    parser.add_argument('--algo_params', type=str, default='all_algos', help='which parameters to use')
    parser.add_argument('--output_filename', type=str, default='round', help='name of output files')
    parser.add_argument('--save_dir', type=str, default='results_output', help='name of save directory')
    parser.add_argument('--save_specs', type=bool, default=False, help='save the architecture specs')    
    parser.add_argument('--save_fig', type=bool, default=True, help='save the results figure')    

    args = parser.parse_args()
    main(args)

import matplotlib.pyplot as plt
import h5py
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
from matplotlib.ticker import StrMethodFormatter
import os
from utils.model_utils import get_log_path, METRICS
import seaborn as sns
import string
import matplotlib.colors as mcolors
import os
COLORS=list(mcolors.TABLEAU_COLORS)
MARKERS=["o", "v", "s", "*", "x", "P"]

plt.rcParams.update({'font.size': 14})
n_seeds=5

def load_results(args, algorithm, seed):
    alg = get_log_path(args, algorithm, seed, args.gen_batch_size)
    hf = h5py.File("./{}/{}.h5".format(args.result_path, alg), 'r')
    metrics = {}
    for key in METRICS:
        metrics[key] = np.array(hf.get(key)[:])
    return metrics


def get_label_name(name):
    name = name.split("_")[0]
    prefix = "GT" if "GT" in name else ""
    if 'Distill' in name:
        if '-FL' in name:
            name = 'FedDistill' + r'$^+$'
        else:
            name = 'FedDistill'
    elif 'FedDF' in name:
        name = 'FedFusion'
    elif 'FedEnsemble' in name:
        name = 'Ensemble'
    elif 'FedAvg' in name:
        name = 'FedAvg'
    return prefix+name

def plot_results(args, algorithms):
    n_seeds = args.times
    dataset_ = args.dataset.split('-')
    sub_dir = dataset_[0] + "/" + dataset_[2] # e.g. Mnist/ratio0.5
    os.system("mkdir -p figs/{}".format(sub_dir))  # e.g. figs/Mnist/ratio0.5
    plt.figure(1, figsize=(5, 5))
    TOP_N = 1
    max_acc = 0
    for i, algorithm in enumerate(algorithms):
        algo_name = algorithm#get_label_name(algorithm)
        ######### plot test accuracy ############
        metrics = [load_results(args, algorithm, seed) for seed in range(n_seeds)]
        all_curves = np.concatenate([np.maximum.accumulate(metrics[seed]['glob_acc'][:args.num_glob_iters]) for seed in range(n_seeds)])
        #all_curves = np.concatenate([metrics[seed]['glob_acc'][:args.num_glob_iters] for seed in range(n_seeds)])
        top_accs =  np.concatenate([np.sort(metrics[seed]['glob_acc'][:args.num_glob_iters])[-TOP_N:] for seed in range(n_seeds)])
        acc_avg = np.mean(top_accs)
        acc_std = np.std(top_accs)
        info = '{}, {}, {}, {:.4f}, {:.4f}'.format(args.result_path, args.num_glob_iters, algo_name, acc_avg , acc_std)
        print(info)
        fname = os.path.join(args.result_path, args.result_path+"_"+str(args.num_glob_iters)+"_"+algorithm+'.csv')
        f = open(fname, "w+")
        print(info, file=f)
        f.close()
        length = len(all_curves) // n_seeds
        ls = '--' if i % 2 ==0 else '-'

        ax=sns.lineplot(
            x=np.array(list(range(length)) * n_seeds) + 1,
            y=all_curves.astype(float),
            legend='brief',
            color=COLORS[i],
            label=algo_name,
            linestyle=ls,
            ci="sd",
            linewidth= 2
        )
    
      
    plt.gcf()
    plt.grid()
    plt.title(dataset_[0] + ' Test Accuracy')
    plt.xlabel('FL Round')
    #plt.legend(ncol = 3)
    max_acc = np.max([max_acc, np.max(all_curves) ])

    if args.min_acc < 0:
        alpha = 3 / 4
        min_acc = np.max(all_curves) * alpha #+ np.min(all_curves) * (1-alpha)
    else:
        min_acc = args.min_acc
    step = np.round((max_acc- min_acc)/8,2)
    plt.yticks(np.round(np.arange(min_acc, max_acc+1e-2, step),2))
    plt.xticks(range(0,201,25))
    plt.ylim(min_acc-1e-2, max_acc+1e-2)
    algs_str = "_Vs_".join(algorithms)
    fig_save_path = os.path.join(args.result_path,args.result_path+"_"+str(args.num_glob_iters)+"_"+algs_str+'.pdf')#os.path.join('figs', sub_dir, dataset_[0] + '-' + dataset_[2] + '.png')
    plt.savefig(fig_save_path, bbox_inches='tight', pad_inches=0.05, format='pdf', dpi=600)
    print('file saved to {}'.format(fig_save_path))
    AX = plt.gca()
    y=[]
    if args.erir > 0:
        for l in AX.lines:
            y.append(l.get_ydata())
        print(len(y))
        y = np.array(y)
        epsilon = args.erir_epsilon
        i = args.erir_round
        tx = np.where(y[0] >= y[0][i]-epsilon)[0][0]
        ty = np.where(y[1] >= y[0][i]-epsilon)[0][0]

        f = open("erir.csv", "a+")
        print("{},{}, {}, {}, {}, {}, {}, {}".format(args.result_path,args.algorithms,
                i, epsilon,y[0][i], tx, ty, 1-ty/tx), file = f)
        f.close() 
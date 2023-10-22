# from asyncio.windows_events import NULL
import numpy as np
import itertools
import pickle
import matplotlib.pyplot as plt

from games.game import Game
from wrapper import GameDynamicsWrapper
from wrapper import VariedGame
from dynamics.replicator import Replicator
import scipy.stats
from scipy.spatial.distance import jensenshannon 
import os

get_player_labels = lambda p : "M" + str(p) 

class GTEnsemble(Game):
    
    NUM_ClIENTS = None
    MODELS_EVALS_MATRIX = None
    PLAYER_LABELS = None
    STRATEGY_LABELS = None
    STRATEGY_PROFILES = None
    
    def __call__(self, **game_kwargs):
        return self         
    
    def __init__(self, num_clients, models_eval_matrix,equilibrium_tolerance=0.2):                 
        self.NUM_ClIENTS = num_clients
        self.MODELS_EVALS_MATRIX = models_eval_matrix              
    
        self.PLAYER_LABELS = tuple([get_player_labels(i) for i in range(self.NUM_ClIENTS)])
    
        self.STRATEGY_LABELS = [] 
        self.STRATEGY_PROFILES = []
        for i in range(self.NUM_ClIENTS):
            self.STRATEGY_LABELS.append([])
            self.STRATEGY_PROFILES.append([])
            for j in range(self.NUM_ClIENTS):
                if i!=j :
                    temp = 'M'+str(j)
                    self.STRATEGY_LABELS[-1].append(temp)
                    self.STRATEGY_PROFILES[-1].append(j)
        self.STRATEGY_LABELS = tuple(tuple(sub) for sub in self.STRATEGY_LABELS)   
        player_dist = tuple([1/(self.NUM_ClIENTS)] * (self.NUM_ClIENTS))
        super(GTEnsemble, self).__init__(payoff_matrices= self.getpay(),
            player_frequencies=player_dist, equilibrium_tolerance=equilibrium_tolerance)
   

    def getpay(self):
        payoff_shape = tuple([self.NUM_ClIENTS]) + (self.NUM_ClIENTS-1,)*(self.NUM_ClIENTS)
        payoff_matrix = np.ones((self.NUM_ClIENTS, (self.NUM_ClIENTS-1)**self.NUM_ClIENTS))
        itr = itertools.product(*self.STRATEGY_PROFILES)
        idx = 0
        for strategy_profile in itr:
            s, c = np.unique(strategy_profile, return_counts=True)
            w = np.zeros(self.NUM_ClIENTS)
            np.add.at(w, s, c/(self.NUM_ClIENTS))
            for p in range(self.NUM_ClIENTS):
                avg = np.average(self.MODELS_EVALS_MATRIX[p,:], axis=0, weights = w)
                payoff_matrix[p, idx] = avg
            idx += 1    
        payoff_matrix = np.reshape(payoff_matrix, payoff_shape)
        return payoff_matrix

def ensemble(num_clients, models_eval_matrix, r, num_iterations, num_gens):
    my_rc_param = {'text.usetex': False}
    plt.rcParams.update(my_rc_param)
    g = GTEnsemble(num_clients= num_clients, models_eval_matrix= models_eval_matrix)
    s = GameDynamicsWrapper(g, Replicator)
    (_,savg,_) = s.simulate_many(num_iterations=num_iterations, num_gens=num_gens, pop_size=num_clients*100,
                             graph=False, return_labeled=False) #dict(options=['smallfont'])
    res = np.array([savg[i][-1] for i in range(num_clients)]).tolist()
    
    for i in range(num_clients):
        res[i] = np.insert(res[i], i, 0)
    
    GTEnsemble_weights = np.average(res, axis=0)/100
    return (GTEnsemble_weights, res)    


def gtflat_say(str):
    print("="*100)
    print(f">> GTFLAT: {str}")
    print("="*100)

def KL(P,Q):
    epsilon = 0.00001

    # You may want to instead make copies to avoid changing the np arrays.
    P = P+epsilon
    Q = Q+epsilon

    divergence = np.sum(P*np.log(P/Q))
    return divergence


def JSD(p, q):
    """
    method to compute the Jenson-Shannon Distance 
    between two probability distributions
    """

    # convert the vectors into numpy arrays in case that they aren't
    p = np.array(p)
    q = np.array(q)

    # calculate m
    m = (p + q) / 2

    # compute Jensen Shannon Divergence
    divergence = (scipy.stats.entropy(p, m) + scipy.stats.entropy(q, m)) / 2

    # compute the Jensen Shannon Distance
    distance = np.sqrt(divergence)

    return distance    

from dynamics.dynamics import DynamicsSimulator
import numpy as np

class Replicator(DynamicsSimulator):
    """
    A stochastic dynamics simulator which performs replicator dynamics on all player types in the population.
    """
    def __init__(self, generation_skip=1, *args, **kwargs):#Recommended to use a higher population number if utilizing generationSkip
        """
        The constructor for the Replicator dynamics process, that the number of births/deaths to precess per time step.
        
        @param generation_skip: to be interpreted as the number of time-steps included in each simulated generation. The larger it is the faster the dynamics.
        @type generation_skip: float

        """
        super(Replicator, self).__init__(*args,stochastic=False,**kwargs)
        self.generation_skip = generation_skip
        
    def next_generation(self, previous_state, group_selection, rate):
        next_state = []
        number_groups=len(previous_state)
        payoff = []
        avg_payoffs = []
        fitness = []
        for i in range(number_groups):
            p, avg_p = self.calculate_payoffs(previous_state[i])
            payoff.append(p)
            avg_payoffs.append(avg_p)
            fitness.append(self.calculate_fitnesses(payoff[i], self.selection_strengthI))
            
        for i in range(number_groups):
            new_group_state =[]
            for pIndex, (fitnesses, stratDistribution, numPlayers) in enumerate(zip(fitness[i], previous_state[i], self.num_players)):
                meanFitness = np.mean(fitnesses)
            
                new_player_state = np.zeros(len(fitnesses))
                for stratIndex, (stratFitness, stratProportion) in enumerate(zip(fitnesses, stratDistribution)):
                    dStrat = stratProportion * (stratFitness - meanFitness) * self.generation_skip
                    new_player_state[stratIndex] = stratProportion + dStrat
                
                for i, strat in enumerate(new_player_state):
                    if strat < 0:
                        new_player_state[i] = 0
                if new_player_state.sum() <= 0:
                    for i in range(len(new_player_state)):
                        new_player_state[i] = 1#Normalization in edge cases (to prevent negative distributions or all 0 distributions)
                    
                new_player_state *= float(numPlayers / new_player_state.sum())
                new_player_state = np.array(self.round_individuals(new_player_state))
                new_group_state.append(new_player_state)
            next_state.append(new_group_state)

        return next_state, fitness
        
    
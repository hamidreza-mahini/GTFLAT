__author__ = 'eblubin@mit.edu, anande01@g.harvard.edu'
import numpy as np
from dynamics.dynamics import DynamicsSimulator


class WrightFisher(DynamicsSimulator):
    def __init__(self, mu = None, *args, **kwargs):
        """
        @param mu: mutation rate
        @type mu: float or a list of n lists, where n is the number of players and len(list_n) = number of player_n strategies. Defaults to zero if not specified.
        """

        # TODO don't allow pop_size of 0, wright fisher only works with finite pop size
        super(WrightFisher, self).__init__(*args,stochastic=True,**kwargs)

        if mu == None:
            mu = 0.0
        self.mu = mu

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

        # Creating the mutation matrix
        if type(self.mu) == float:
            mu_matrix = []
            for i in range(len(payoff[0])):
                mu_matrix.append(self.mu*np.ones(len(payoff[0][i])))
        else:
            mu_matrix = self.mu

        # Wright-Fisher between groups
        if group_selection and np.random.uniform(0,1)<rate:

            avg_fitness = []

            # Calculate the fitness of each group based on their average payoffs
            for k in range(len(avg_payoffs)):
                avg_fitness.append(self.fitness_func(avg_payoffs[k], self.selection_strengthG))
            # Groups reproduce proportional to their fitness

            new_group_distribution = np.random.multinomial(number_groups,[x / sum(avg_fitness) for x in avg_fitness])


            # Update the new distribution of groups
            for idx, group_freq in enumerate(new_group_distribution):
                for i in range(group_freq):
                    next_state.append(previous_state[idx])

        # Wright-Fisher inside groups
        else:
            for i in range(number_groups):
                new_group_state =[]
                for player_idx, (strategy_distribution, fitnesses, num_players) in enumerate(zip(previous_state[i], fitness[i], self.num_players)):
                    num_strats = len(strategy_distribution)
                    total_mutations = 0
                    new_player_state = np.zeros(num_strats)

                    for strategy_idx, n in enumerate(strategy_distribution):
                        f = fitnesses[strategy_idx]
                        mu_individual = mu_matrix[player_idx][strategy_idx]

                        # sample from binomial distribution to get number of mutations for strategy
                        if n == 0:
                            mutations = 0
                        else:
                            mutations = np.random.binomial(n, mu_individual)
                        n -= mutations
                        total_mutations += mutations
                        new_player_state[strategy_idx] = n * f

                        # distribute player strategies proportional n * f
                        # don't use multinomial, because that adds randomness we don't want yet.

                    if new_player_state.sum() != 0:
                        new_player_state *= float(num_players - total_mutations) / new_player_state.sum()
                        new_player_state = np.array(self.round_individuals(new_player_state))

                    else: # Make sure that mutations get randomly distributed if they lead to a zero population size
                        new_player_state = np.zeros(num_strats)

                    new_player_state += np.random.multinomial(total_mutations, [1. / num_strats] * num_strats)
                    new_group_state.append(new_player_state)
                next_state.append(new_group_state)

        return next_state, fitness

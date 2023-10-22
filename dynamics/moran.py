from dynamics.dynamics import DynamicsSimulator
import numpy as np

class Moran(DynamicsSimulator):
    """
    A stochastic dynamics simulator that performs the Moran process on all player types in the population.
    See U{Moran Process<http://en.wikipedia.org/wiki/Moran_process#Selection>}
    """
    def __init__(self, mu = None, *args, **kwargs):
        """
        The constructor for the Moran dynamics process, that the number of births/deaths to process per time step.
        Mutations are incorporated at the individual level for now.
        # TO DO: Variable iterations per time step
        @param mu: mutation rate
        @type mu: float or a list of n lists, where n is the number of players and len(list_n) = number of player_n strategies. Defaults to zero if not specified.
        """
        super(Moran, self).__init__(*args,stochastic=True,**kwargs)
        if mu == None:
            mu = 0.0
        self.mu = mu

    def next_generation(self, previous_state, group_selection, rate):
        next_state = []

        # Copy to the new state
        for p in previous_state:
            next_state.append(p.copy())

        number_groups = len(previous_state)
        payoff = []
        avg_payoffs = []
        fitness = []
        for i in range(number_groups):
            p, avg_p = self.calculate_payoffs(previous_state[i])
            payoff.append(p)
            avg_payoffs.append(avg_p)
            fitness.append(self.calculate_fitnesses(payoff[i], self.selection_strengthI))

        total_fitness_per_player_type = [[] for i in range(len(previous_state[0]))] # This length could be written better.
        for i in range(len(previous_state[0])):
            for j in range(len(previous_state)):
                for k in range(len(fitness[j][i])):
                    total_fitness_per_player_type[i].append(fitness[j][i][k]*next_state[j][i][k])

        # Creating the mutation matrix
        if type(self.mu) == float:
            mu_matrix = []
            for i in range(len(payoff[0])):
                mu_matrix.append(self.mu*np.ones(len(payoff[0][i])))
        else:
            mu_matrix = self.mu

        # Moran at the group level
        if group_selection and np.random.uniform(0,1)<rate:

            avg_fitness = []

            # Calculate the fitness of each group based on their average payoffs
            for k in range(len(avg_payoffs)):
                avg_fitness.append(self.fitness_func(avg_payoffs[k], self.selection_strengthG))

            # Pick the group that will reproduce and the one that it replaces
            reproduction = np.random.multinomial(1,[x / sum(avg_fitness) for x in avg_fitness])
            reproduction_index = np.nonzero(reproduction)[0][0]
            replacement_event = np.random.randint(0,number_groups)
            next_state[replacement_event] = next_state[reproduction_index]
        else:

            # Moran at individual level where one individual from all the groups is chosen to reproduce proportional to it's fitness
            group = []
            strategy = []

            # For each player-type pick one individual from one group to reproduce
            for i in range(len(total_fitness_per_player_type)):
                weighted_total = sum(total_fitness_per_player_type[i])
                dist = np.array([f_i/weighted_total for f_i in total_fitness_per_player_type[i]])
                sample = np.random.multinomial(1,dist)
                reproduce_index = np.nonzero(sample)[0][0]
                player_strat = len(total_fitness_per_player_type[i])/number_groups
                group.append(int(reproduce_index/player_strat))
                strategy.append(int(reproduce_index%player_strat))

            # Pick a random individual to replace from the same group as the reproducing individual
            for player_no, (group_no,strat_no) in enumerate(zip(group,strategy)):
                p = next_state[group_no][player_no]
                mu_individual = mu_matrix[player_no][strat_no]

                # Determine who dies
                total = p.sum()
                dist = [n_i / float(total) for n_i in p]

                # Chance of mutating while reproduction
                if np.random.uniform(0,1)<mu_individual:
                    strat_no = np.random.randint(0,len(p))
                p[strat_no] += 1
                p -= np.random.multinomial(1, dist)
            next_state[group_no][player_no] = p

        return next_state, fitness

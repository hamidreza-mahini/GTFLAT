from plot import plot_data_for_players, GraphOptions
from results import NDimensionalData
import inspect
import numpy as np
import types
import marshal
from parallel import par_for, delayed, wrapper_simulate, wrapper_vary_for_kwargs
from util import Obj
from graphSetup import setupGraph, setupHistogram

DEFAULT_ITERATIONS = 100  #: The default number of iterations for which to run a repeated simulation
DEFAULT_GENERATIONS = 300  #: the default number of generations for which to run a simulation


class GameDynamicsWrapper(object):
    """
    A helper class that wraps a dynamics class and a game class and provides helper methods for simulation.
    """
    def __init__(self, game_cls, dynamics_cls, game_kwargs=None, dynamics_kwargs=None):
        """
        Initialize the wrapper with a subclass of Game and DynamicsSimulator, and optional keyword arguments that
        override the defaults
        @param game_cls: the game to wrap
        @type game_cls: L{Game}.__class__
        @param dynamics_cls: the type of dynamics to use
        @type dynamics_cls: L{DynamicsSimulator}.__class__
        @param game_kwargs: any keyword arguments that will be passed to the game class on initialization
        @type game_kwargs: dict
        @param dynamics_kwargs: any keyword arguments that will be passed to the dynamics class on initialization
        @type dynamics_kwargs: dict

        """
        self.game_kwargs = game_cls.DEFAULT_PARAMS
        if game_kwargs is not None:
            self.game_kwargs.update(game_kwargs)
        if dynamics_kwargs is None:
            dynamics_kwargs = {}
        self.game_cls = game_cls
        self.dynamics_cls = dynamics_cls
        self.dynamics_kwargs = dynamics_kwargs


    def update_game_kwargs(self, *args, **kwargs):
        """
        Update the default values of the arguments to the game constructor for a VariedGame.
        @param args: dictionary(s) to update the values with
        @type args: dict
        @param kwargs: keys of the dictionary to update.
        """
        self.game_kwargs.update(*args, **kwargs)

    def update_dynamics_kwargs(self, *args, **kwargs):
        """
        Update the default values of the arguments to the dynamics constructor for a VariedGame.
        @param args: dictionary(s) to update the values with
        @type args: dict
        @param kwargs: keys of the dictionary to update.
        """
        self.dynamics_kwargs.update(*args, **kwargs)

    def stationaryDistribution(self):
        pass



    def simulate(self, num_gens=DEFAULT_GENERATIONS, pop_size=100, start_state=None, graph=True, return_labeled=True, burn=0, class_end=False, frac_invasions = False, strategy_indx = 0):
        """
        Simulate a game in the presence of group selection for a specific number of generations optionally
        graphing the results

        @param num_gens: the number of iterations of the simulation.
        @type num_gens: int
        @param group_size: Fixed population in each group.
        @type group_size: int
        @param rate: Rate of group selection vs individual selection
        @type rate: float
        @param start_state: whether the starting state is to be predetermined
        @type start_state: list
        @param graph: the type of graph (false if no graph is wished)
        @type graph: dict, bool
        @param return_labeled: whether the distribution of classified equilibria that are returned should be labelled
            or simply listed with their keys inferred by their order
        @type return_labeled: bool
        @param class_end: if False the equilibria are classified at all generations and if True only the last generation is classified.
        @type class_end: bool
        @param frac_invasions: Whether the given simulation is to compute the fraction of invasions for a strategy.
        @type: bool
        @param strategy_indx: The index of the strategy whose fraction of invasions is to be computed
        @type strategy_indx: int
        @return: the frequency of time spent in each equilibria, defined by the game
        @rtype: numpy.ndarray or dict
        TO DO: Explain what 'burn' does
        """

        game = self.game_cls(**self.game_kwargs)
        dyn = self.dynamics_cls(payoff_matrix=game.pm,
                                player_frequencies=game.player_frequencies,pop_size=pop_size,
                                **self.dynamics_kwargs)

        # Group Selection simulation for a given number of generations.
        results_total,payoffs_total=dyn.simulate(num_gens,start_state,frac_invasions,strategy_indx)

        # Classify the equilibria and plot the results
        params = Obj(**self.game_kwargs)

        frequencies = np.zeros(self.game_cls.num_equilibria())  # one extra for the Unclassified key
        if dyn.stochastic:
            classifications = []

            if class_end:  # Only classify the final generation
                lastGenerationState = [np.zeros(len(player[0])) for player in results_total]
                for playerIdx, player in enumerate(results_total):
                    for stratIdx, strat in enumerate(player[-1]):
                        lastGenerationState[playerIdx][stratIdx] = strat
                    lastGenerationState[playerIdx] /= lastGenerationState[playerIdx].sum()
                equi = game.classify(params, lastGenerationState, game.equilibrium_tolerance)
                frequencies = np.zeros(self.game_cls.num_equilibria())
                frequencies[equi] = 1

            else: # Classify every generation

                for state in zip(*results_total):
                    state = [x / x.sum() for x in state]
                    equi = game.classify(params, state, game.equilibrium_tolerance)
                    # note, if equi returns -1, then the -1 index gets the last entry in the array
                    classifications.append(equi)
                    frequencies[equi] += 1
        else:
            last_generation_state = [results_total[-1][-1]]
            classification = game.classify(params, last_generation_state, game.equilibrium_tolerance)
            frequencies[classification] = 1
        
        if graph:
            setupGraph(graph, game, dyn, burn, num_gens, results_total, payoffs_total)
        else:
            if return_labeled:
                return self._convert_equilibria_frequencies(frequencies)
            else:
                return frequencies, results_total, payoffs_total

    # Add ways to plot the evolution of strategies in a single group?

    def simulate_many(self, num_iterations=DEFAULT_ITERATIONS, num_gens=DEFAULT_GENERATIONS, pop_size=100, start_state=None, graph=False, histogram = False, return_labeled=True, burn=0, parallelize=True, class_end=True, frac_invasions = False, strategy_indx = 0):
        """
        A helper method to call the simulate methods num_iterations times simulating num_gens generations each time,
        and then averaging the frequency of the resulting equilibria. Method calls are parallelized and attempt to
        use all available cores on the machine.

        @param num_iterations: the number of times to iterate the simulation
        @type num_iterations: int
        @param num_gens: the number of generations to run each simulation with
        @type num_gens: int
        @param pop_size: total population size
        @type pop_size: int
        @param start_state: An optional list of distributions of strategies for each player
        @type start_state: list or None
        @param graph: the type of graph (false if no graph is wished)
        @type graph: dict, bool
        @param histogram: if True plots the histogram of final population sizes over iterations. Both graph and histogram can't be True at the same time.
        @type histogram: bool
        @param return_labeled: whether the distribution of classified equilibria that are returned should be labelled
            or simply listed with their keys inferred by their order
        @type return_labeled: bool
        @param parallelize: whether or not to parallelize the computation, defaults to true, but an override when
            varying the parameters, as seen in the L{VariedGame} class to achieve coarser parallelization
        @type parallelize: bool
        @param class_end: if False the equilibria are classified at all generations and if True only the last generation is classified.
        @type class_end: bool
        @param frac_invasions: Whether the given simulation is to compute the fraction of invasions for a strategy.
        @type: bool
        @param strategy_indx: The index of the strategy whose fraction of invasions is to be computed
        @type strategy_indx: int
        @return: the frequency of time spent in each equilibria, defined by the game
        @rtype: np.ndarray or dict
        """
        # TODO move this graphing into graphSetup and link it to extra options

        game = self.game_cls(**self.game_kwargs)
        dyn = self.dynamics_cls(payoff_matrix=game.pm,
                                player_frequencies=game.player_frequencies,pop_size=pop_size,
                                **self.dynamics_kwargs)
        frequencies = np.zeros(self.game_cls.num_equilibria())

        output = par_for(parallelize)(delayed(wrapper_simulate)(self, num_gens=num_gens, pop_size=pop_size, frac_invasions = frac_invasions, strategy_indx = strategy_indx, start_state= start_state, class_end=class_end) for iteration in range(num_iterations))

        equilibria = []
        strategies = [0]*num_iterations
        payoffs = [0]*num_iterations
        for idx, sim in enumerate(output):
            equilibria.append(sim[0])
            strategies[idx] = sim[1]
            payoffs[idx] = sim[2]

        #TODO move these averages or only compile them if appropriate simulation type
        stratAvg = [np.zeros(shape=(num_gens, dyn.pm.num_strats[playerIdx])) for playerIdx in range(dyn.pm.num_player_types)]
        # Storing the final strategy populations per iteration
        strat_final = [np.zeros(shape=(num_iterations, dyn.pm.num_strats[playerIdx])) for playerIdx in range(dyn.pm.num_player_types)]

        for iteration in range(num_iterations):
            for player in range(dyn.pm.num_player_types):
                for gen in range(num_gens - burn):
                    for strat in range(dyn.pm.num_strats[player]):
                        stratAvg[player][gen][strat] += strategies[iteration][player][gen][strat]
                        if histogram and gen == num_gens - burn - 1:
                            strat_final[player][iteration][strat] += strategies[iteration][player][gen][strat]

        for playerIdx, player in enumerate(stratAvg):
            for genIdx, gen in enumerate(player):
                if gen.sum() != 0:
                    gen /= gen.sum()
                    gen *= dyn.num_players[playerIdx]

        payoffsAvg = [np.zeros(shape=(num_gens -1, dyn.pm.num_strats[playerIdx])) for playerIdx in range(dyn.pm.num_player_types)]
        for iteration in range(num_iterations):
            for player in range(dyn.pm.num_player_types):
                for gen in range(num_gens -1 - burn):
                    for strat in range(dyn.pm.num_strats[player]):
                        payoffsAvg[player][gen][strat] += payoffs[iteration][player][gen][strat]

        for playerIdx, player in enumerate(payoffsAvg):
            for genIdx, gen in enumerate(player):
                if gen.sum() != 0:
                    gen /= gen.sum()
                    gen *= dyn.num_players[playerIdx]

        if graph:
            assert histogram == False, ("Can't plot graph and histogram at the same time")
            setupGraph(graph, game, dyn, burn, num_gens, stratAvg, payoffs[0])
        elif histogram:
            setupHistogram(histogram, game, dyn, num_iterations, dyn.num_players, strat_final)

        for x in equilibria:
            frequencies += x

        frequencies /= frequencies.sum()    
        if return_labeled:
            return self._convert_equilibria_frequencies(frequencies)
        else:
            return frequencies, stratAvg, strat_final

    def frac_invasions(self, strategy_indx, num_iterations=1000, num_gens=1000, pop_size=100, parallelize = True):
        """
        Calculates the fraction of runs where a given strategy dominates by the end of the simulation as defined by the equilibrium_tolerance
        in a symmetric game. It runs the simulate method num_iterations times, where the start state consists of one player from the strategy
        of interest in the population of other player strategies. This approximates the fixation probability for large enough iterations and
        number of generations when the equilibrium_tolerance is 0.

        @param strategy_indx: The index of the strategy whose fraction of invasions is to be computed
        @type strategy_indx: int
        @param num_iterations: Number of iterations to run of the simulation.
        @type: int
        @param num_gens: Number of generations to run for each simulation.
        @type: int
        @param pop_size: Size of the population.
        @type: int
        @return: Fraction of runs where the required strategy dominated the population.
        @type: str
        """
        strategies = np.asarray(self.game_cls.STRATEGY_LABELS)
        assert strategies.ndim == 1, ("Works for only symmetric games.")

        frac = self.simulate_many(num_iterations = num_iterations, num_gens = num_gens, pop_size = pop_size, frac_invasions = True, strategy_indx = strategy_indx, return_labeled = False, parallelize=parallelize)
        return ('Fraction of runs where the required strategy dominated the population = %0.2f' %frac[strategy_indx])

    @staticmethod
    def _static_convert_equilibria_frequencies(game_cls, frequencies):
        labels = game_cls.get_equilibria()
        return {label: freq for label, freq in zip(labels, frequencies) if freq > 0}

    def _convert_equilibria_frequencies(self, frequencies):
        """
        Convert the list of frequencies of equilibria to a dictionary mapping equilibrium name to frequency
        """
        return self._static_convert_equilibria_frequencies(self.game_cls, frequencies)


class IndependentParameter(object):
    """
    A class that encapsulates the notion of a parameter that varies from simulation to simulation
    """
    def __init__(self, lb, ub, num_steps):
        """
        Construct an independent parameter. A varied simulation can have one or more independent parameters. Each
        independent parameter has a lower bound, an upper bound, the number of steps, a unique key identifying / labelling it
        (usually its param name for the game), a boolean indicating whether or not it is a direct input to the constructor,
        and a boolean indicating whether the parameter is for the dynamics constructor or game constructor.

        @param lb: The lower bound of the variation
        @type lb: int or float
        @param ub: The upper bound of the variation
        @type ub: int or float
        @param num_steps: the number of steps in between the lower bound and upper bound
        @type num_steps: int
        """
        self.lb = float(lb)
        self.ub = float(ub)
        self.num_steps = num_steps

    def _step_size(self):
        return (self.ub - self.lb) / self.num_steps

    def __getitem__(self, item):
        if item >= 0 and item <= self.num_steps:
            return self.lb + item * self._step_size()
        elif item < 0 and item >= -self.num_steps - 1:
            # negative indexing starts from the back
            return self.ub + (1 + item) * self._step_size()
        else:
            raise IndexError

    def __len__(self):
        return self.num_steps + 1


class VerboseIndependentParameter(IndependentParameter):
    """
    An extension on the IndependentParameter class that makes room for three other properties
        - The key that the independent parameter is varying
        - Whether or not the parameter is direct or indirect (indirect may be used as params to dependent params, but don't directly get applied to the class constructor)
        - Whether the parameter is for the dynamics or the class constructor
    """
    def __init__(self, key, is_game_kwarg, is_direct, *args, **kwargs):
        self.key = key
        self.is_direct = is_direct
        self.is_game_kwarg = is_game_kwarg
        super(VerboseIndependentParameter, self).__init__(*args, **kwargs)


class DependentParameter(object):
    """
    A dependent parameter that is defined as a function of the other parameters in the simulation.
    """
    def __init__(self, func):
        """
        Each dependent parameter can be a function of both the values of all the other parameters, as well as,
        any other inputs. Due to the fact that in order to parallelize we need to be able to pickle all the arguments,
        the lambda function is easier to pickle without any closure. As a result the lambda cannot reference any
        external variables, besides those passed in as arguments.

        @param func: the function mapping fixed parameters to the value that this dependent paramter should take on
        @type func: lambda
        """
        assert func.func_closure is None, "In order to support parallelization, the lambda must NOT be a closure. It can only be a function of the parameters to the simulation."
        self.func = func

    def get_val(self, **kwargs):
        """
        Evaluate the dependent parameter as a function of the other parameters for the namespace.

        @param kwargs: the parameters to be used as input to the dependent variable
        """
        return self.func(Obj(**kwargs))

    # Hack to allow portability of lambdas cross-process.
    # The only requirement is that the function doesn't have a closure, which we check above
    def __getstate__(self):
        return {'func': marshal.dumps(self.func.func_code)}

    def __setstate__(self, state):
        self.func = types.FunctionType(marshal.loads(state['func']), globals())

class VariedGame(object):
    """
    A class that wraps the L{GameDynamicsWrapper} class and simplifies the process of varying multiple parameters
    to the simulation at once, and then graphing the effect one or more parameters have on the resulting equilibrium
    distribution of repeated simulations.
    """
    def __init__(self, game_cls, dynamics_cls, game_kwargs=None, dynamics_kwargs=None):
        """
        Initialize the wrapper with a subclass of Game and DynamicsSimulator, and optional keyword arguments that
        override the defaults
        @param game_cls: the game to wrap
        @type game_cls: L{Game}.__class__
        @param dynamics_cls: the type of dynamics to use
        @type dynamics_cls: L{DynamicsSimulator}.__class__
        @param game_kwargs: any keyword arguments that will be passed to the game class on initialization
        @type game_kwargs: dict
        @param dynamics_kwargs: any keyword arguments that will be passed to the dynamics class on initialization
        @type dynamics_kwargs: dict
        """
        self.game_cls = game_cls
        self.game_kwargs = game_kwargs if game_kwargs is not None else {}
        self.dynamics_cls = dynamics_cls
        self.dynamics_kwargs = dynamics_kwargs if dynamics_kwargs is not None else {}

    def vary_param(self, kw, low_high_num_steps, **kwargs):
        low, high, num_steps = low_high_num_steps
        """
        A helper function to vary one parameter of the game instance over a range of values, and graph the results

        @param kw: the keyword to vary
        @type kw: str
        @param low: the lower limit (inclusive) of the variation
        @type low: float or int
        @param high: the upper limit (inclusive) of the variation
        @type high: float or int
        @param num_steps: the number of steps to break the variation into. 1 indicates two total simulations, one at
            the lower limit and one at the upper limit. Must be larger than one
        @type num_steps: int
        @rtype: L{NDimensionalData}
        @return: the data for the parameter variation for all different values.
        """
        if 'graph' not in kwargs:
            kwargs['graph'] = dict(type='2d')

        return self.vary(game_kwargs={kw: (low, high, num_steps)}, **kwargs)

    def vary_2params(self, kw1, low1_high1_num_steps1, kw2, low2_high2_num_steps2, **kwargs):
        low1, high1, num_steps1 = low1_high1_num_steps1
        low2, high2, num_steps2 = low2_high2_num_steps2#Unpacking the tuples
        """
        A helper function to vary two parameters of the game instance over an independent range of values, and graph the results.

        @param kw1: the keyword to vary
        @type kw1: str
        @param low1: the lower limit (inclusive) of the first variation
        @type low1: float or int
        @param high1: the upper limit (inclusive) of the first variation
        @type high1: float or int
        @param num_steps1: the number of steps to break the first variation into. 1 indicates two total simulations, one at
            the lower limit and one at the upper limit. Must be larger than one
        @type num_steps1: int
        @param kw2: the lower keyword to vary
        @type kw2: str
        @param low2: the lower limit (inclusive) of the second variation
        @type low2: float or int
        @param high2: the upper limit (inclusive) of the second variation
        @type high2: float or int
        @param num_steps2: the number of steps to break the second variation into. 1 indicates two total simulations, one at
            the lower limit and one at the upper limit. Must be larger than one
        @type num_steps2: int
        @rtype: L{NDimensionalData}
        @return: the data for the parameter variation for all different values.
        """

        if 'graph' not in kwargs:
            kwargs['graph'] = dict(type='3d')

        return self.vary(game_kwargs={kw1: (low1, high1, num_steps1), kw2: (low2, high2, num_steps2)}, **kwargs)

    def vary(self, game_kwargs=None, dynamics_kwargs=None, num_iterations=DEFAULT_ITERATIONS, num_gens=DEFAULT_GENERATIONS, burn=0, graph=False, parallelize=True):
        """
        We can vary the game kwargs, the dynamics kwargs, as well as any number of indirect inputs, if needed
        Each of these parameters must be an iterable of dictionaries, in the following form:

        game_kwargs = [{INDEPENDENT},{DEPENDENT}, {INDIRECT}]
        INDEPENDENT:
        Each key must be the string of the param name, as seen in the constructor
        Each value is an iterable of 3 values (lower_bound, upper_bound, num_steps)

        DEPENDENT:
        Each key must be the string of the param name, as seen in the constructor, cannot have any of the keys in the
        keys of the INDEPENDENT dict
        Each value is a function that takes in kwargs for the namespace

        INDIRECT:
        Each key must be the string of the param name, as seen in the constructor, cannot have any of the keys in the
        keys of the INDEPENDENT or DEPENDENT dicts


        If the root item is actually a dictionary, and not a list/tuple, then there are assumed to be no dependent kwargs or INDIRECT
        """
        assert not (game_kwargs is None and dynamics_kwargs is None), "nothing to vary!"

        kwargs = [game_kwargs, dynamics_kwargs]

        for j, kw in enumerate(kwargs):
            if kw is None:
                kwargs[j] = [{}, {}, {}]
            else:
                if isinstance(kw, dict):
                    kwargs[j] = [kw, {}, {}]
                else:
                    assert isinstance(kw, (list, tuple))
                    if len(kw) == 2:
                        kw.append({})
                    assert len(kw) == 3
                    # verify no duplicate keys
                    key_set = set()

                    for d in kw:
                        for k in d:
                            assert k not in key_set
                            key_set.add(k)

        assert len(kwargs[0][0]) > 0 or len(kwargs[1][0]) > 0 or len(kwargs[1][2]) > 0 or len(kwargs[0][2]) > 0, "We don't actually have any parameters to iterate over"

        independent_params = []
        for i, kw in enumerate(kwargs):
            for j in (0, 2):
                for k in kw[j]:
                    v = kw[j][k]
                    assert len(v) == 3
                    ip = VerboseIndependentParameter(k, i == 0, j == 0, *v)
                    kw[j][k] = ip
                    independent_params.append(ip)
            assert isinstance(kw[1], dict)
            for k in kw[1]:
                v = kw[1][k]
                argspec = inspect.getargspec(v)
                assert len(argspec.args) == 1
                kw[1][k] = DependentParameter(v)

        w = GameDynamicsWrapper(self.game_cls, self.dynamics_cls, self.game_kwargs, self.dynamics_kwargs)

        dependent_params = (kwargs[0][1], kwargs[1][1])
        results = self._vary_kwargs(independent_params, dependent_params, w, num_iterations=num_iterations, num_gens=num_gens, burn=burn, parallelize=parallelize)

        data = NDimensionalData.initialize(results, independent_params)

        # TODO: persist results
        if graph:
            data.graph(self.game_cls.get_equilibria(), graph)

        return data

    def _vary_kwargs(self, ips, dependent_params, sim_wrapper, **kwargs):
        return self._vary_for_kwargs(ips, 0, dependent_params, sim_wrapper, (), **kwargs)

    def _vary_for_kwargs(self, ips, idx, dependent_params, sim_wrapper, chosen_vals, parallelize=False, **kwargs):
        """
        A recursively defined function to iterate over all possible permutations of the variables defined in the list
        of independent variables that returns the simulation results of the cross product of these variable variations.

        @param ips: a list of all the VerboseIndependentParameters that will be varied
        @type ips: list(L{VerboseIndependentParameter})
        @param idx: the index of the independent parameter about to be iterated upon
        @type idx: int
        @param dependent_params: the tuple of dictionaries representing the DependentParameters for the game_kwargs
            and the dynamics_kwargs, respectively.
        @type dependent_params: tuple({string: DependentParameter})
        @param sim_wrapper: the pre-initialized sim-wrapper on which we will call simulate_many
        @type sim_wrapper: L{GameDynamicsWrapper}
        @param chosen_vals: a tuple of all the indices of the chosen values for each already-decided independent param
        @type chosen_vals: tuple(int)
        @param parallelize: whether or not to parallelize the subloops of this function. We set to true on the parent call
            and then false for all recursive calls.
        @type parallelize: bool
        @param kwargs: These are the rest of the keyword arguments that should be passed directly to the simulate_many function call
        @rtype: list(list(...))
        @return: a recursive list of lists representing the simulation results for having assigned each independent parameter
            the value corresponding to the index at which the simulation results are present in the list of lists.

            i.e. Two independent parameters, the return type will be a list of lists of simulation results

            The simulation result present at the address [4][17] represents the value of the simulation when the first
            independent parameter was set to its value at index 4 (@see IndependentParameter.__getitem__), and the second
            independent parameter was set to its value at index 17.
        """
        if idx == len(ips):
            # the list is divided as follows:
            # [[direct_game_kwargs, indirect_game_kwargs], [direct_dynamics_kwargs, indirect_dynamics_kwargs]]
            varied_kwargs = [[{}, {}], [{}, {}]]

            # helper function to return the correct keywords give the desired params
            def kws(ip):
                return varied_kwargs[int(not ip.is_game_kwarg)][int(not ip.is_direct)]

            for chosen_idx, ip in zip(chosen_vals, ips):
                kws(ip)[ip.key] = ip[chosen_idx]

            # the list is organized as follows:
            # [game_kwargs, dynamics_kwargs]
            sim_kwargs = [{}, {}]

            for i in (0, 1):
                # set all the direct ones
                for k, v in varied_kwargs[i][0].items():
                    sim_kwargs[i][k] = v

                # now calculate all of the dependent parameters, as a function of both direct
                # and indirect independent parameters
                for k, dp in dependent_params[i].items():
                    # get the inputs to the dependent param calculation
                    if i == 0:

                        dependent_kw_params = sim_wrapper.game_kwargs.copy()
                    else:
                        dependent_kw_params = sim_wrapper.dynamics_kwargs.copy()
                    dependent_kw_params.update(varied_kwargs[i][0])
                    dependent_kw_params.update(varied_kwargs[i][1])

                    sim_kwargs[i][k] = dp.get_val(**dependent_kw_params)

            sim_wrapper.update_dynamics_kwargs(sim_kwargs[1])
            sim_wrapper.update_game_kwargs(sim_kwargs[0])
            # don't parallellize the simulate_many requests, we are parallelizing higher up in the call chain
            return sim_wrapper.simulate_many(return_labeled=False, parallelize=False, **kwargs)

        var_indices = range(len(ips[idx]))
        #dependent_params = [{}, {}]
        return par_for(parallelize)(delayed(wrapper_vary_for_kwargs)(self, ips, idx + 1, dependent_params, sim_wrapper, chosen_vals + (i, ), **kwargs) for i in var_indices)

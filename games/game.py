__author__ = 'elubin'

from payoff_matrix import PayoffMatrix
from util import Obj
import numpy
import itertools
import time
import tempfile
import os
import logging
from parallel import par_for, delayed
import multiprocessing

UNCLASSIFIED_EQUILIBRIUM = 'Unclassified'  #: the string used to identify an equilibrium that did not match any of the classification rules

class Game(object):
    """
    A class that is used to encapsulate the notion of a game theory game. Each game is identified by a set number of
    players, each choosing from a pre-determined set of strategies, as well as the logic the defines the equilibria
    for the game.
    """
    DEFAULT_PARAMS = {}  #: the default parameters that are passed into the constructor by the L{GameDynamicsWrapper}
    PLAYER_LABELS = None  #: a list of labels to apply to each player in the game, used in graphing
    STRATEGY_LABELS = None  #: a list of lists of strings that name the available strategies for each player
    EQUILIBRIA_LABELS = ()  #: a list of labels corresponding to the integers returned by the classify function


    def __init__(self, payoff_matrices, player_frequencies, bias_strength = 0, bias_scale = 0, equilibrium_tolerance=0.1):
        """
        Initializes the game class with the give list of payoff matrices and distribution of players, as well as
        a notion of the equilibrium tolerance.

        @param payoff_matrices: a list of recursive lists representing the payoff matrix for each player, see L{PayoffMatrix} for more info
        @type payoff_matrices: list
        @param player_frequencies: a list that describes the distribution of players by player type, must sum to 1
        @type player_frequencies: list or tuple
        @param bias_strength: Relative strength of conformity versus individual selection
        @type bias_strength: float between 0 and 1
        @param bias_scale: The payoff associated with conformity, it can be thought of as the payoff associated with a coordination game.
        @type bias_scale: float
        @param equilibrium_tolerance: the flexibility that should be used for equilibrium classification. An
            equilibrium is classified as such if 1 - equilibrium_tolerance proportion of people are playing a given
            set of strategies
        @type equilibrium_tolerance: float
        """
        assert payoff_matrices is not None
        assert player_frequencies is not None
        if self.PLAYER_LABELS is not None:
            assert len(player_frequencies) == len(self.PLAYER_LABELS)

        self.pm = PayoffMatrix(len(player_frequencies), payoff_matrices, bias_strength, bias_scale)
        if self.STRATEGY_LABELS is not None:
            for labels_i, num_strats  in zip(self.STRATEGY_LABELS, self.pm.num_strats):
                assert len(labels_i) == num_strats

        self.player_frequencies = player_frequencies
        self.equilibrium_tolerance = equilibrium_tolerance

    @classmethod
    def classify(cls, params, state, tolerance):
        """
        A class method that should be override by subclasses to help classify equilibria as a function of the current
        state of the population, the parameters to the game instance's constructor, and the equilibrium tolerance.

        @param params: An object that encapsulates all the parameters passed in to this object's constructor.
        @type params: L{Obj}
        @param state: a list of lists representing the distribution of players in each state
        @type state: list(list())
        @param tolerance: the equilibrium tolerance with which the instance was constructed
        @type tolerance: float
        @return: an integer representing the index of the equilibrium to which the state corresponds.
        @rtype: int
        """
        return -1

    @classmethod
    def num_equilibria(cls):
        """
        Get the number of equilibria for the game. This is one more than the number defined by the user.
        @return: the # of equilibria
        @rtype: int
        """
        return len(cls.EQUILIBRIA_LABELS) + 1

    @classmethod
    def get_equilibria(cls):
        """
        Get the list of equilibria defined by the class, plus the string representing the unclassified equlibrium, which
        can be easily accessed by indexing -1 on the tuple.

        @return: a tuple of strings of the equilibrium labels
        @rtype: tuple
        """
        return tuple(cls.EQUILIBRIA_LABELS) + (UNCLASSIFIED_EQUILIBRIUM, )


    # TODO: distribute on 4 cores
    @classmethod
    def validate_classifier(cls, timeout=None, tolerance=0.05, **kwargs):
        game_kwargs = cls.DEFAULT_PARAMS
        game_kwargs.update(kwargs)
        g = cls(**game_kwargs)
        params = Obj(**game_kwargs)

        def generate_state_from_pure_strategy(p_idx, n_strategies):
            s = numpy.zeros((n_strategies, ))
            s[p_idx] = 1.0
            return s

        def convert_state(s):
            return [{cls.STRATEGY_LABELS[j][i]: s_i[i] for i in range(len(s_i)) if s_i[i] > 0} for j, s_i in enumerate(s)]

        false_negatives = []
        false_positives = []

        n_players = g.pm.num_player_types
        # 1. first validate all pure strategy equilibria (non mixed) by iterating through all permutations of all strategies
        for perm in g.pm.get_all_strategy_tuples():
            assert len(perm) == n_players
            state = []
            for i, s in enumerate(perm):
                state.append(generate_state_from_pure_strategy(s, g.pm.num_strats[i]))

            eq = cls.classify(params, state, tolerance)
            is_eq = g.pm.is_pure_equilibrium(perm)
            if is_eq == True:
                if eq == -1:
                    false_negatives.append(state)
            else:
                profitable_deviation = is_eq
                if eq != -1:
                    false_positives.append((state, eq, profitable_deviation))

        def print_results(false_negatives, false_positives, kwargs, num_sims):
            output = StringIO.StringIO()
            print >>output, 'Parameters: %s' % kwargs
            print >>output, 'Total states tried: %d' % num_sims
            print >>output, "# False negatives: %d" % len(false_negatives)
            print >>output, "# False positives: %d" % len(false_positives)
            if len(false_negatives) > 0:
                print >>output, "False negatives:"

                for fn in false_negatives:
                    print >>output, convert_state(fn)

            if len(false_positives) > 0:
                print >>output, "False positives:"
                for state, eq, p_dev in false_positives:
                    first = 'State: %s, Classification: %s. ' % (convert_state(state), cls.EQUILIBRIA_LABELS[eq])
                    if p_dev[0]:
                        # mixed strategy deviation
                        second = 'Profitable deviation: player %d - strategies (%s:%f, %s:%f) don\'t have same exp payoff' % (p_dev[1], cls.STRATEGY_LABELS[p_dev[1]][p_dev[2][0][0]], p_dev[2][0][1], cls.STRATEGY_LABELS[p_dev[1]][p_dev[2][1][0]], p_dev[2][1][1])
                    else:
                        # pure strategy deviation
                        second = 'Profitable deviation: player %d - %s' % (p_dev[1], cls.STRATEGY_LABELS[p_dev[1]][p_dev[2]])
                    print >>output, first + second
            fd, name = tempfile.mkstemp(suffix='.txt')
            os.write(fd, output.getvalue())
            os.close(fd)
            return name

        def generate_strat_mixins(n_strats, p_i, prefix):
            if len(prefix) == n_strats:
                yield prefix
                return
            if (p_i, len(prefix)) in g.pm.dominated_strategies:
                choices = [False]
            else:
                choices = [False, True]

            for c in choices:
                for mix in generate_strat_mixins(n_strats, p_i, prefix + (c, )):
                    if any(mix):
                        yield mix




        # for all players, generate all possible mixes of available strategies that are not dominated strategies
        strategy_permutations = []
        for p_i in range(n_players):
            n_strats = g.pm.num_strats[p_i]
            strategy_permutations.append(list(generate_strat_mixins(n_strats, p_i, ())))

        product = itertools.product(*strategy_permutations)
        product = list(product)
        # TODO: filter out any mixes over strategies that have the same payoff for all players
        def mix_over_strategies(s_tuple):
            n = sum(int(x) for x in s_tuple)
            r = numpy.random.dirichlet([1] * n)
            mix_idx = 0
            state_partition = numpy.zeros((len(s_tuple, )))
            for i, should_mix in enumerate(s_tuple):
                if should_mix:
                    state_partition[i] = r[mix_idx]
                    mix_idx += 1
            return state_partition

        # start the timer
        start = time.time()

        # while the timer hasn't run out
        ATTEMPTS_PER_PERMUTATION = 10

        def should_end():
            if timeout is None:
                return False
            else:
                return time.time() - start > timeout

        def do_work():
            permutations = 0
            while not should_end():
                for i in range(ATTEMPTS_PER_PERMUTATION):
                    for perm in product:
                        permutations += 1
                        # if none are mixed strategies, then skip
                        mixed = False
                        for s_tuple in perm:
                            n = sum(int(x) for x in s_tuple)
                            if n > 1:
                                mixed = True

                        if not mixed:
                            continue

                        state = [mix_over_strategies(player_strats) for player_strats in perm]

                        eq = cls.classify(params, state, tolerance)
                        is_eq = g.pm.is_mixed_equilibrium(state)
                        if is_eq == True:
                            if eq == -1:
                                false_negatives.append(state)
                        else:
                            profitable_deviation = is_eq
                            if eq != -1:
                                false_positives.append((state, eq, profitable_deviation))

            return permutations
        num_sims = do_work()



        output_file = print_results(false_negatives, false_positives, game_kwargs, num_sims)
        print ('Saved results to file: %s' % output_file)
        #par_for()(delayed(do_work)() for _ in range(multiprocessing.cpu_count()))


# common case is n = 2, but we support as big N as needed
class SymmetricNPlayerGame(Game):
    """
    A convenience class that provides the logic for an N player game where each player chooses from the same strategy
    set.
    """
    def __init__(self, payoff_matrix, n, bias_strength = 0, bias_scale = 0, equilibrium_tolerance=0.1):
        """
        Initialize the symmetric game with the given payoff matrix and number of players

        @param payoff_matrix: a recursive list representing the payoff matrix for each player, see L{PayoffMatrix}
        @type payoff_matrix: list(list())
        @param n: the number of player types in the game which is generally 1 for a 2 player symmetric game.
        @type n: int
        """
        if self.STRATEGY_LABELS is not None:
            self.STRATEGY_LABELS = (self.STRATEGY_LABELS, ) * n

        # TODO: append as many as specified by num_players! not just one more
        # TODO: just need to think how to "transpose" a multidimensional matrix

        # interpreted as multiple instances of the same player, append the transpose
        payoff_matrix_2 = tuple(map(tuple, zip(*payoff_matrix))) # transpose
        matrices = [payoff_matrix,payoff_matrix_2] 
        player_dist = (1, ) # Works only for n=1 player type
        super(SymmetricNPlayerGame, self).__init__(payoff_matrices=matrices, player_frequencies=player_dist, bias_strength=bias_strength, bias_scale = bias_scale, equilibrium_tolerance=equilibrium_tolerance)

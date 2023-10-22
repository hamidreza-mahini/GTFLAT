from plot import plot_data_for_players, GraphOptions, plotHistogram


def setupGraph(graph, game=None, dyn=None, burn=None, num_gens=None, results=None, payoffs=None):  # TODO allow ordering of various lines
    if graph is True:
        graph = dict()
    graph_options = graph
    if 'options' in graph_options:
        for key in graph_options['options']:
            graph_options[key] = True
        del graph_options['options']

    yPos = 0

    if game is not None and game.STRATEGY_LABELS is not None:
        graph_options[GraphOptions.LEGEND_LABELS_KEY] = lambda p, s: game.STRATEGY_LABELS[p][s]

    if game is not None and game.PLAYER_LABELS is not None:
        graph_options[GraphOptions.TITLE_KEY] = lambda p: game.PLAYER_LABELS[p]

    if any(k in graph_options for k in ['payoffLine', 'modeStratLine', 'meanStratLine']):
        yPos = 0
        graph_options['colorLineArray'] = [[] for player in results]
        graph_options['textList'] = []

    if 'payoffLine' in graph_options:
        yPos -= 0.05
        for playerIdx, player in enumerate(payoffs):
            colorLineArray = []
            for gen in range(burn, num_gens - 1):
                maxPayoff = 0
                maxPayoffIdx = -1
                for payoffIdx, payoff in enumerate(player[gen]):
                    if payoff > maxPayoff:
                        maxPayoff = payoff
                        maxPayoffIdx = payoffIdx

                currentGen = gen - burn
                nextGen = gen - burn + 1

                line = [currentGen, nextGen, yPos, yPos]
                colorLineArray.append([line, maxPayoffIdx])
            # colorLineArray[0][1] = colorLineArray[1][1]  # To fill in first gen
            graph_options['colorLineArray'][playerIdx].extend(colorLineArray)
            #graph_options['textList'].append(([-num_gens / 7, yPos], 'Best Strat'))  # TODO fix x positioning

    if 'modeStratLine' in graph_options:
        yPos -= 0.05
        for playerIdx, player in enumerate(results):
            colorLineArray = []
            for gen in range(burn, num_gens):
                maxStratProp = 0
                maxStratIdx = -1
                for stratIdx, stratProp in enumerate(player[gen]):
                    if stratProp > maxStratProp:
                        maxStratProp = stratProp
                        maxStratIdx = stratIdx

                currentGen = gen - burn
                nextGen = gen - burn + 1
                line = [currentGen, nextGen, yPos, yPos]
                colorLineArray.append([line, maxStratIdx])
            graph_options['colorLineArray'][playerIdx].extend(colorLineArray)
            #graph_options['textList'].append(([-num_gens / 7, yPos], 'Modal Strat'))

    if 'meanStratLine' in graph_options:
        yPos -= 0.05
        #graph_options['textList'].append(([-num_gens / 7, yPos], 'Mean Strat'))

    yPos -= 0.025

    graph_options[GraphOptions.NO_MARKERS_KEY] = True

    if results is not None:
        plot_data_for_players(results, range(burn, num_gens), "Generation #", dyn.pm.num_strats,
                          num_players=dyn.num_players,
                          graph_options=graph_options, yBot=yPos)

    if 'graph_payoffs' in graph_options:
        if burn == 0:
            burn = 1
        plot_data_for_players(payoffs, range(burn, num_gens), "Generation #", dyn.pm.num_strats,
                              num_players=dyn.num_players,
                              graph_options=dict(), title="Normalized Payoffs")

def setupHistogram(histogram, game = None, dyn = None, num_iterations = None, num_players = None, results = None):

    if histogram is True:
        histogram = dict()
    graph_options = histogram
    if 'options' in graph_options:
        for key in graph_options['options']:
            graph_options[key] = True
        del graph_options['options']

    if game is not None and game.STRATEGY_LABELS is not None:
        graph_options[GraphOptions.LEGEND_LABELS_KEY] = lambda p, s: game.STRATEGY_LABELS[p][s]

    if game is not None and game.PLAYER_LABELS is not None:
        graph_options[GraphOptions.TITLE_KEY] = lambda p: game.PLAYER_LABELS[p]

# For each player plotting the histogram of final population of each strategy
    for playerIdx in range(dyn.pm.num_player_types):
        strat_iterations = [[] for j in range(dyn.pm.num_strats[playerIdx])]
        for strat in range(dyn.pm.num_strats[playerIdx]):
            for iteration in range(num_iterations):
                strat_iterations[strat].append(results[playerIdx][iteration][strat])
        xmax = num_players[playerIdx]
        plotHistogram(strat_iterations, xmax, "Population size", playerIdx, num_strats = dyn.pm.num_strats[playerIdx], graph_options=graph_options)

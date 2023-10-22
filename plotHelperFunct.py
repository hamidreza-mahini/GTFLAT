from matplotlib import colors

def graphLines(lineArray, plt):
    for line in lineArray:
        plt.plot([line[0], line[1]], [line[2], line[3]], 'k-', lw=1)


def graphColoredLines(lineArray, plt, colors):  # List of lines and associated colors
    for (line, color) in lineArray:
        if type(color) is list:
            plt.plot([line[0], line[1]], [line[2], line[3]], color=tuple(color), lw=15)
        else:
            plt.plot([line[0], line[1]], [line[2], line[3]], color=colors[color], lw=15)


def colorAvg(colorList, proportionList):
    newColor = [0, 0, 0]
    for color, prop in zip(colorList, proportionList):
        if type(color) is str:
            color = colors.colorConverter.to_rgb(color)
        for idx, value in enumerate(color):
            newColor[idx] += value * prop

    return newColor


def stackProportions(data):  # Turns proportional data into total data
    for generation in data:
        for i, cat in enumerate(generation):
            if not i == 0:
                generation[i] += generation[i-1]


def normalize(value_range, normalizeTo=1):  # Normalizes data to 1, useful for type proportions
    for i, step in enumerate(value_range):
        value_range[i] = step / (step + normalizeTo)
    return value_range

def plotText(textList, plt, fontsize):
    for position, text in textList:
        plt.text(*position, text, fontsize=fontsize)
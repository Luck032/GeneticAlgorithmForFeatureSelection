import pandas as pd
import numpy as np
import random
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from deap import creator, base, tools, algorithms
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
import sys


random.seed(42)
np.random.seed(42)

def avg(l):
    """
    Returns the average between list elements
    """
    return (sum(l)/float(len(l)))


def getFitness(individual, X, y):
    """
    Feature subset fitness function
    """

    if(individual.count(0) != len(individual)):
        # get index with value 0
        cols = [index for index in range(
            len(individual)) if individual[index] == 0]

        # get features subset
        X_parsed = X.drop(X.columns[cols], axis=1)
        X_subset = pd.get_dummies(X_parsed)

        # apply classification algorithm
        clf = RandomForestClassifier(random_state=42, n_jobs=-1)

        return (avg(cross_val_score(clf, X_subset, y, cv=cv)),)
    else:
        return(0,)


def geneticAlgorithm(X, y, n_population, n_generation):
    """
    Deap global variables
    Initialize variables to use eaSimple
    """
    # create individual
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    # create toolbox
    toolbox = base.Toolbox()
    toolbox.register("attr_bool", random.randint, 0, 1)
    toolbox.register("individual", tools.initRepeat,
                     creator.Individual, toolbox.attr_bool, len(X.columns))
    toolbox.register("population", tools.initRepeat, list,
                     toolbox.individual)
    toolbox.register("evaluate", getFitness, X=X, y=y)
    toolbox.register("mate", tools.cxOnePoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # initialize parameters
    pop = toolbox.population(n=n_population)
    hof = tools.HallOfFame(n_population * n_generation)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)

    # genetic algorithm
    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.1,
                                   ngen=n_generation, stats=stats, halloffame=hof,
                                   verbose=False)

    # return hall of fame
    return hof


def bestIndividual(hof, X, y):
    """
    Get the best individual
    """
    maxAccuracy = 0.0
    _individual = None

    for individual in hof:
        acc = individual.fitness.values[0]
        if acc > maxAccuracy:
            maxAccuracy = acc
            _individual = individual

    _individualHeader = [list(X)[i] for i in range(len(_individual)) if _individual[i] == 1]
    return maxAccuracy, _individual, _individualHeader


if __name__ == '__main__':


    # Load your data

    # X = ...
    # y = ...

    cv = StratifiedKFold(n_splits=3)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, stratify=y, random_state=42)

    individual = [1 for i in range(len(X_train.columns))]
    print("Accuracy with all features: \t" +
            str(getFitness(individual, X_train, y_train)) + "\n")

    n_gen = 50
    n_pop = 50

    hof = geneticAlgorithm(X_train, y_train, n_pop, n_gen)

    accuracy, individual, header = bestIndividual(hof, X_train, y_train)
    print('Number of Features in Subset: \t' + str(individual.count(1)))
    print('Feature Subset\t: ' + str(header))

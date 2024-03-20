# Szymon Dyszewski
import matplotlib.pyplot as plt
import numpy as np
import os.path


class Plot:
    def __init__(self, LR, DF, E) -> None:
        self.mutation_strategies = None
        self.mutations = None
        self.discrete_f = None
        self.fs = None
        self.distances = None
        self.distances_from_optimum = None
        self.epizods = None
        self.percentages = None
        self.path = f'./charts/LR: {LR}, DF: {DF}, E: {E}/'
        if not os.path.isdir(self.path):
            os.makedirs(self.path)

    def report(self):
        self.drawPlotMutationTypes()
        self.drawPlotFValues()
        self.drawDistances()
        self.drawPercentages()
        self.write()

    def drawPlotMutationTypes(self):
        fig = plt.figure(figsize=(10, 5))

        # creating the bar plot
        plt.bar(self.mutation_strategies, self.mutations, color='green',
                width=0.4)

        plt.xlabel("Type of mutation")
        plt.ylabel("No. of choices")
        plt.title("Average mutations usage:")
        plt.savefig(f'{self.path}MutationTypes')

    def drawPlotFValues(self):
        fig = plt.figure(figsize=(10, 5))
        # creating the bar plot
        plt.bar(self.discrete_f, self.fs, color='blue',
                width=0.05)

        plt.xlabel("F value")
        plt.ylabel("No. of choices")
        plt.title("Average F usage:")
        plt.savefig(f'{self.path}PlotFValues')

    def drawDistances(self):
        fig = plt.figure(figsize=(10, 5))
        plt.locator_params(axis='x', nbins=len(self.epizods))
        # creating the bar plot
        plt.plot(self.epizods, self.distances, color='red', marker='o')

        plt.xlabel("No. of epizod:")
        plt.ylabel("Distance in population:")
        plt.title("Distance in population over epizods")
        plt.savefig(f'{self.path}AverageDistance')

    def drawPercentages(self):
        fig = plt.figure(figsize=(10, 5))
        plt.locator_params(axis='x', nbins=len(self.epizods))
        # creating the bar plot
        plt.plot(self.epizods, self.percentages, color='black', marker='o')

        plt.xlabel("No. of epizod:")
        plt.ylabel("Percentage of better children:")
        plt.title("Percentage of better children over epizods")
        plt.savefig(f'{self.path}AveragePercentage')

    def write(self):
        with open(f'{self.path}data.txt', "w") as file:
            file.write(f'Average population distance from optiumum: {np.average(self.distances_from_optimum)}\n')
            file.write(f'STD population distance from optiumum: {np.std(self.distances_from_optimum)}\n')
            file.write(f'Min population distance from optiumum: {np.min(self.distances_from_optimum)}\n')
            file.write(f'Max population distance from optiumum: {np.max(self.distances_from_optimum)}\n')

# Szymon Dyszewski
import numpy as np
from cec2017.functions import f1, f3
np.random.seed(0)


class DE():
    def __init__(self, population: np.ndarray, function=1) -> None:
        if function == 1:
            self.function = f1
        else:
            self.function = f3
        self.population = population
        self.evaluation = self.function(population)
        self.prev_evaluation = self.evaluation
        self.better_counter = 0

    def nextGeneration(self, F: float, CR: float, mutation_strategy: str) -> None:
        best_val = 0
        best = 0
        for individual in self.population:
            if best_val > self.function([individual]):
                best_val = self.function([individual])
                best = individual
        for id, individual in enumerate(self.population):
            mutant_vector = self.mutate(F, mutation_strategy, best)
            recombined_individual = self.recombination(mutant_vector, individual, CR)
            self.selection(recombined_individual, id)
        return None

    def mutate(self, F: float, mutation_strategy: str, best: np.ndarray) -> np.ndarray:
        match mutation_strategy:
            case "rand/1":
                """
                Performs the 'rand/1' mutation strategy in Differential Evolution.
                """
                r1, r2, r3 = np.random.choice(len(self.population), size=3, replace=False)
                mutant_vector = self.population[r1] + F * (self.population[r2] - self.population[r3])
            case "best/1":
                """
                Performs the 'best/1' mutation strategy in Differential Evolution.
                """
                r1, r2 = np.random.choice(len(self.population), size=2, replace=False)
                mutant_vector = best + F * (self.population[r1] - self.population[r2])
            case "rand/2":
                """
                Performs the 'rand/2' mutation strategy in Differential Evolution.
                """
                r1, r2, r3, r4, r5 = np.random.choice(len(self.population), size=5, replace=False)
                mutant_vector = self.population[r1] + F * (self.population[r2] - self.population[r3]) \
                    + F * (self.population[r4] - self.population[r5])
            case "best/2":
                """
                Performs the 'best/2' mutation strategy in Differential Evolution.
                """
                r1, r2, r3, r4 = np.random.choice(len(self.population), size=4, replace=False)
                mutant_vector = best + F * (self.population[r1] - self.population[r2]) \
                    + F * (self.population[r3] - self.population[r4])
        return mutant_vector

    def recombination(self, mutator: np.ndarray, individual: np.ndarray, CR: float) -> np.ndarray:
        recombined_individual = np.array(individual)
        for attribute_id, attribute in enumerate(mutator):
            random_treshold = np.random.random()
            if random_treshold < CR:
                recombined_individual[attribute_id] = attribute
        return recombined_individual

    def selection(self, recombined_individual: np.ndarray, id: int) -> None:
        if self.function([recombined_individual]) < self.function([self.population[id]]):
            self.population[id] = recombined_individual
            self.better_counter += 1
        return None

    def evaluateChildren(self) -> float:
        percentage = self.better_counter/len(self.population)*100
        self.better_counter = 0
        return percentage

    def evaluateOptimum(self) -> float:
        summarized_diffrence = 0
        for individual in self.population:
            summarized_diffrence += abs(self.function([individual]) - 100)
        return summarized_diffrence / len(self.population)

    def evaluateDistanceInPopulation(self) -> float:
        avg_attribute = [0 for _ in self.population[0]]
        for individual in self.population:
            for attribute in individual:
                avg_attribute += attribute
        for attribute in self.population[0]:
            avg_attribute /= len(self.population)

        distance_in_population = 0
        for individual in self.population:
            for attributeid, attribute in enumerate(individual):
                distance_in_population += abs(attribute - avg_attribute[attributeid])
        distance_in_population /= len(self.population)
        return distance_in_population

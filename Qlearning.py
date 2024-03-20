# Szymon Dyszewski
import numpy as np
from charts import Plot
from DE import DE
np.random.seed(0)


class QL:
    def __init__(self, population, CR=0.5, learning_rate=0.1, discount_factor=0.4,
                 num_episodes=25, max_steps=1000) -> None:
        self.CR = CR
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.num_episodes = num_episodes
        self.max_steps = max_steps
        self.population = population
        self.chartData = Plot(learning_rate, discount_factor, num_episodes)

    def calculate(self):
        # Initialize the Q-values
        mutation_strategies = ["rand/1", "best/1", "rand/2", "best/2"]

        # Define the mapping of the distance from the optimum, percentage of better children
        state_mapping = {}
        discrete_distance = [200*x for x in range(10)]
        discrete_percentage = [5*x for x in range(20)]
        idx = 0
        for dist in discrete_distance:
            for percentage in discrete_percentage:
                state_mapping[(dist, percentage)] = idx
                idx += 1

        # Define the mapping of the actions to the f values, strategies
        action_mapping = {}
        discrete_f = [x/20 for x in range(2, 7)]  # f - <0.1, 0.3>
        discrete_strategies = [x for x in range(len(mutation_strategies))]
        idx = 0
        for f_value in discrete_f:
            for strategy in discrete_strategies:
                action_mapping[(f_value, strategy)] = idx
                idx += 1

        # Define variables to compare results over episodes
        distances = []
        distances_from_optimum = []
        children = []
        fs = [0 for _ in discrete_f]
        strategies = [0 for _ in mutation_strategies]
        fs_map = {}
        for _ in discrete_f:
            fs_map[_] = 0
        # Q-learning algorithm
        Q_values = np.zeros((len(state_mapping), len(action_mapping)))
        for episode in range(self.num_episodes):
            # Reset the environment and get the initial state
            f_value = 0.2
            strategy = np.random.randint(0, 4)
            differential_evolution = DE(np.array(self.population))
            average_percentage_of_better_children = 0

            for step in range(self.max_steps):
                # Perform the DE optimization using the current F parameter, mutation strategy, and value

                # Compute the reward based on the performance of the DE optimization
                differential_evolution.nextGeneration(f_value, self.CR, mutation_strategies[strategy])
                distance_in_population = differential_evolution.evaluateDistanceInPopulation()
                percentage_of_better_children = differential_evolution.evaluateChildren()

                # Set a reward function
                reward = percentage_of_better_children

                # Discretize values
                f_idx = min(discrete_f, key=lambda x: abs(x-f_value))
                distance_in_population_idx = min(discrete_distance, key=lambda x: abs(x-distance_in_population))
                percentage_of_better_children_idx = min(discrete_percentage, key=lambda x: abs(x-percentage_of_better_children))

                # Update the Q-value of the current state-action pair
                # Find best action for the current state in the Q value
                # Find current state id
                optimal_state = (distance_in_population_idx, percentage_of_better_children_idx)
                optimal_state_id = state_mapping[optimal_state]
                optimal_action_value = -1e12

                # Explore
                if step < self.max_steps * 0.2:
                    next_f_value = np.random.choice(discrete_f)
                    next_strategy = np.random.random_integers(0, 3)
                # Find best action_id in the current state
                else:
                    for idx in action_mapping.values():
                        if optimal_action_value <= Q_values[optimal_state_id, idx]:
                            optimal_action_value = Q_values[optimal_state_id, idx]
                            optimal_action_idx = idx
                    # Select the optimal F parameter and mutation strategy based on learned Q-values
                    for action in action_mapping:
                        if action_mapping[action] == optimal_action_idx:
                            next_f_value = action[0]
                            next_strategy = action[1]

                next_f_idx = min(discrete_f, key=lambda x: abs(x-next_f_value))

                state = (distance_in_population_idx, percentage_of_better_children_idx)
                action = (f_idx, strategy)
                next_action = (next_f_idx, next_strategy)

                state_id = state_mapping[state]
                action_id = action_mapping[action]
                next_action_id = action_mapping[next_action]

                fs_map[f_idx] += 1
                strategies[strategy] += 1

                # Update the Q-value of the current state-action pair
                Q_values[state_id, action_id] = (1 - self.learning_rate) * Q_values[state_id, action_id] + \
                    self.learning_rate * (reward + self.discount_factor * (Q_values[state_id, next_action_id] - Q_values[state_id, action_id]))

                # Transition to the next state
                f_value = next_f_value
                strategy = next_strategy
                average_percentage_of_better_children += percentage_of_better_children

            average_percentage_of_better_children /= self.max_steps
            current_pop_distance = differential_evolution.evaluateDistanceInPopulation()
            current_distance_from_optimum = differential_evolution.evaluateOptimum()
            distances.append(current_pop_distance)
            distances_from_optimum.append(current_distance_from_optimum)
            children.append(average_percentage_of_better_children)
            fs.append(next_f_value)
            strategies.append(next_strategy)
            print(f'{episode + 1} / {self.num_episodes}, {(episode + 1) / self.num_episodes * 100}%')

        # Transfer data to charts
        for x in range(len(mutation_strategies)):
            strategies[x] /= self.num_episodes
        for x in fs_map:
            fs_map[x] /= self.num_episodes
        self.chartData.mutation_strategies = mutation_strategies
        self.chartData.mutations = strategies[:len(mutation_strategies)]
        self.chartData.discrete_f = discrete_f
        self.chartData.fs = fs_map.values()
        self.chartData.distances = distances
        self.chartData.distances_from_optimum = distances_from_optimum
        self.chartData.percentages = children
        self.chartData.epizods = [x + 1 for x in range(self.num_episodes)]
        self.chartData.report()

# Szymon Dyszewski
from Qlearning import QL
# from DE import DE
import numpy as np
np.random.seed(0)

UPPER_BOUND = 100
DIMENSIONALITY = 30
POPULATION = 100
population = np.random.uniform(-UPPER_BOUND, UPPER_BOUND, size=(POPULATION, DIMENSIONALITY))
CR = 0.4
LR = [0.1, 0.3, 0.5, 0.7]
DF = [0.4]
NUMBEROFEPISODES = [30]
NUMBEROFSTEPS = 1000

# Manual look for good parameters
# mutation_strategies = ["rand/1", "best/1", "rand/2", "best/2"]
# for f in [x/10 for x in range(1, 4)]:
#     for cr in [x/10 for x in range(3, 8)]:
#         print(f, cr)
#         avg = 0
#         for _ in range(0, 10):
#             de = DE(np.array(population))
#             for _ in range(NUMBEROFSTEPS):
#                 de.nextGeneration(f, cr, np.random.choice(mutation_strategies))
#             avg += de.evaluateOptimum()
#         print(avg / 10)

# Look for good parameters
# for lr in LR:
#     for df in DF:
#         for episodes in NUMBEROFEPISODES:
#             ql = QL(np.array(population), CR, lr, df, episodes, NUMBEROFSTEPS)
#             ql.train()

ql = QL(np.array(population), CR, 0.7, 0.4, 30, 1000)
ql.calculate()

ql = QL(np.array(population), CR, 0.3, 0.4, 30, 1000)
ql.calculate()

ql = QL(np.array(population), CR, 0.7, 0.7, 30, 1000)
ql.calculate()

ql = QL(np.array(population), CR, 0.3, 0.7, 30, 1000)
ql.calculate()

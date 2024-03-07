import numpy as np
from pymoo.algorithms.base.genetic import GeneticAlgorithm
from pymoo.algorithms.moo.nsga2 import binary_tournament
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.selection.tournament import TournamentSelection
from pymoo.operators.survival.rank_and_crowding import RankAndCrowding
from pymoo.termination.default import DefaultMultiObjectiveTermination
from pymoo.util.misc import has_feasible


class DynamicNSGA2(GeneticAlgorithm):

    def __init__(self,
                 pop_size=40,
                 sampling=FloatRandomSampling(),
                 selection=TournamentSelection(func_comp=binary_tournament),
                 crossover=SBX(eta=15, prob=0.9),
                 mutation=PM(eta=20),
                 survival=RankAndCrowding(),
                 output=None,
                 pc1_max=0.95,
                 pc1_min=0.4,
                 pm1_max=0.1,
                 pm1_min=0.01,
                 **kwargs):

        super().__init__(
            pop_size=pop_size,
            sampling=sampling,
            selection=selection,
            crossover=crossover,
            mutation=mutation,
            survival=survival,
            output=output,
            advance_after_initial_infill=True,
            **kwargs)

        self.termination = DefaultMultiObjectiveTermination()
        self.tournament_type = 'comp_by_dom_and_crowding'
        self.pc1_max = pc1_max
        self.pc1_min = pc1_min
        self.pm1_max = pm1_max
        self.pm1_min = pm1_min

    def _set_optimum(self, **kwargs):
        if not has_feasible(self.pop):
            self.opt = self.pop[[np.argmin(self.pop.get("CV"))]]
        else:
            self.opt = self.pop[self.pop.get("rank") == 0]

    def _next(self):
        super()._next()

        # Update crossover probability
        self.crossover.prob = self.pc1_max - (self.pc1_max - self.pc1_min) * (self.n_gen / self.max_gen)

        # Update mutation probability
        self.mutation.prob = self.pm1_min + (self.pm1_max - self.pm1_min) * (self.n_gen / self.max_gen)

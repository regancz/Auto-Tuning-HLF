from pymoo.algorithms.moo.age import AGEMOEA
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.moo.rnsga2 import RNSGA2
from pymoo.algorithms.moo.sms import SMSEMOA
from pymoo.algorithms.soo.nonconvex.pso_ep import EPPSO
from pymoo.algorithms.soo.nonconvex.sres import SRES
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.util import ref_dirs

from Model.moo.optnsga2 import DynamicNSGA2

# rnsga2 = RNSGA2(
#     ref_dirs,
#     pop_size=40,
#     n_offsprings=10,
#     sampling=FloatRandomSampling(),
#     crossover=SBX(prob=0.9, eta=15),
#     mutation=PM(eta=20),
#     eliminate_duplicates=True
# )

nsga2 = NSGA2(
    pop_size=20,
    n_offsprings=30,
    sampling=FloatRandomSampling(),
    crossover=SBX(prob=0.9, eta=15),
    mutation=PM(eta=20),
    eliminate_duplicates=True
)

# sres = SRES(n_offsprings=200, rule=1.0 / 7.0, gamma=0.85, alpha=0.2)
smsemoa = SMSEMOA()
# nsga3 = NSGA3(pop_size=92,
#               ref_dirs=ref_dirs)
#
# moead = MOEAD(
#     ref_dirs,
#     n_neighbors=15,
#     prob_neighbor_mating=0.7,
# )

# eppso = EPPSO()

agemoea = AGEMOEA(pop_size=40)

dnsga2 = DynamicNSGA2(pop_size=40, pc1_max=0.95, pc1_min=0.4, pm1_max=0.1, pm1_min=0.01)

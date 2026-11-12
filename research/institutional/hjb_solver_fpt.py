"""
Compatibilité : les solveurs vivent désormais dans `research/institutional/hjb_solvers.py`.
`HJBSolverFPT` est le double arrêt optimal actualisé de Leung & Li (2015) ;
`HJBSolverErgodic` (rendement par unité de temps, Bertram 2010) est le défaut du moteur.
"""
from research.institutional.hjb_solvers import HJBSolverErgodic, HJBSolverFPT, make_solver  # noqa: F401

if __name__ == "__main__":
    solver = HJBSolverFPT(theta=0.5, mu=0.0, sigma=0.02, rf=1e-3, cost=0.005)
    results = solver.solve()
    print(f"Leung-Li : entrée long d*={results['d_star']:.4f} | sortie long b*={results['b_star']:.4f}")

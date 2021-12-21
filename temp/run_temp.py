import sys
sys.path.append('../')

import os
os.environ['CASTLE_BACKEND'] = 'pytorch'
# os.environ['CASTLE_BACKEND'] = 'mindspore'

from castle.common import GraphDAG
from castle.metrics import MetricsDAG
from castle.datasets import load_dataset
from castle.datasets import DAG, IIDSimulation, THPSimulation, Topology

from castle.algorithms import RL
from castle.algorithms import GOLEM
from castle.algorithms import GraNDAG
from castle.algorithms import TTPM


def result(n, true_dag):
    # result
    GraphDAG(n.causal_matrix, true_dag)
    met = MetricsDAG(n.causal_matrix, true_dag)
    print(met.metrics)


if 'IID Test':
    # dataset
    X, true_dag, topology_matrix = load_dataset(name='IID_Test')

    # GOLEM
    g = GOLEM(num_iter=10000, device_type='cpu', device_ids=0)
    g.learn(X, dag=true_dag)
    result(g, true_dag)

    # RL
    n = RL(nb_epoch=2000, device_type='cpu', device_ids=0, seed=1)
    n.learn(X, dag=true_dag)
    result(n, true_dag)

    # GraNDAG
    gr = GraNDAG(input_dim=X.shape[1], iterations=10000)
    gr.learn(X)
    result(gr, true_dag)

if 'THP Test':
    # dataset
    X, true_dag, topology_matrix = load_dataset(name='THP_Test')

    # TTPM
    t = TTPM(topology_matrix, max_hop=2)
    t.learn(X)
    result(t, true_dag)

if 'load real dataset':
    # V24_N439_Microwave
    X, true_dag, topology_matrix = load_dataset(name='V24_N439_Microwave', download=True)


def run(alg='golem', nodes=10, n_fold=1, method='linear', gtype='ER'):

    # dataset
    if method == 'linear' and gtype == 'ER':
        weighted_random_dag = DAG.erdos_renyi(n_nodes=nodes, n_edges=nodes*n_fold, 
                                              weight_range=(0.5, 2.0), seed=1)
        dataset = IIDSimulation(W=weighted_random_dag, n=2000, method=method, sem_type='gauss')
        true_dag, X = dataset.B, dataset.X
    
    # learn
    if alg == 'golem':
        # golem
        n = GOLEM(num_iter=10000)
        n.learn(X)
    elif alg == 'rl':
        # rl
        n = RL(nb_epoch=2000)
        n.learn(X, true_dag=true_dag)
    else:
        raise TypeError(f'Algorithm {alg} not supported')

    print(f'alg: {alg}, nodes: {nodes}, n_fold: {n_fold}, method: {method}, gtype: {gtype}')
    GraphDAG(n.causal_matrix, true_dag)
    met = MetricsDAG(n.causal_matrix, true_dag)
    print(met.metrics)
    
    return n.causal_matrix, true_dag


def testing(alg='golem'):
    
    for method in ['linear', 'nonlinear']:
        for gtype in ['ER', 'SF']:
            for n_fold in [1, 2, 4]:
                for nodes in [10, 20, 50, 100]:
                    _, _ = run(alg, nodes, n_fold, method, gtype)
    print(f'finished testing algorithm: {alg}')


causal_matrix, true_dag = run('golem', nodes=10, n_fold=1, method='linear', gtype='ER')
causal_matrix, true_dag = run('rl', nodes=10, n_fold=1, method='linear', gtype='ER')

testing('golem')
testing('rl')

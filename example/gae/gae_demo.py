# coding=utf-8
# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This demo script aim to demonstrate
how to use GAE algorithm in `castle` package for causal inference.

If you want to plot causal graph, please make sure you have already install
`networkx` package, then like the following import method.

Warnings: This script is used only for demonstration and cannot be directly
          imported.
"""

from castle.common import GraphDAG
from castle.metrics import MetricsDAG
from castle.datasets import DAG, IIDSimulation
from castle.algorithms import GAE


#######################################
# graph_auto_encoder used simulate data
#######################################
# simulate data for graph-auto-encoder
weighted_random_dag = DAG.erdos_renyi(n_nodes=10, n_edges=20, weight_range=(0.5, 2.0), seed=1)
dataset = IIDSimulation(W=weighted_random_dag, n=2000, method='linear', sem_type='gauss')
true_dag, X = dataset.B, dataset.X

# graph_auto_encoder learn
ga = GAE(num_encoder_layers=2, num_decoder_layers=2, hidden_size=16,
         max_iter=20, h_tol=1e-12, iter_step=300, rho_thres=1e20, rho_multiply=10, 
         graph_thres=0.2, l1_graph_penalty=1.0, init_iter=5, use_float64=True)
ga.learn(X)

# plot est_dag and true_dag
GraphDAG(ga.causal_matrix, true_dag)

# calculate accuracy
met = MetricsDAG(ga.causal_matrix, true_dag)
print(met.metrics)

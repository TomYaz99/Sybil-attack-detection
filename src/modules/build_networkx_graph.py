import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import networkx as nx
from typing import Dict
from src.base.mock_network import NetworkNode

def build_networkx_graph(nodes: Dict[str, NetworkNode], use_weights: bool = False) -> nx.DiGraph:
    '''
    Builds a directed graph from the given node dictionary.
    If use_weights is True, edge weights are based on trust scores.
    '''
    graph = nx.DiGraph()
    for node_id, node in nodes.items():
        for neighbor_id in node.get_neighbors():
            weight = node.get_reputation_score(neighbor_id) if use_weights else 1.0
            graph.add_edge(node_id, neighbor_id, weight=weight)
    return graph
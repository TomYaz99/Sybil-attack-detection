import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.modules.trust_score import TrustScoreNetwork, TrustNode
from src.modules.build_networkx_graph import build_networkx_graph
import networkx as nx
import random
from typing import Dict
from collections import defaultdict

class RandomWalkDetection(TrustScoreNetwork):
    def random_walk_visits(self, start_node: str, steps: int = 10000) -> Dict[str, float]:
        """
        Perform a trust-biased random walk using NetworkX from a given start node.
        Returns a dictionary of normalized visit frequencies.
        """
        graph = build_networkx_graph(self.nodes, use_weights=True)
        current = start_node
        visit_counts = defaultdict(int)

        for _ in range(steps):
            visit_counts[current] += 1
            neighbors = list(graph.successors(current))
            if not neighbors:
                break
            weights = [graph[current][n].get("weight", 1.0) for n in neighbors]
            total = sum(weights)
            probabilities = [w / total for w in weights]
            current = random.choices(neighbors, weights=probabilities)[0]

        return {node: count / steps for node, count in visit_counts.items()}



if __name__ == "__main__":
    '''
    Same type of example usage as in trust_score.py:
    python random_walks.py --nodes 10 --sybil 3 --runtime 15
    '''    
    import argparse

    parser = argparse.ArgumentParser(description="Random Walk Sybil Detection")
    parser.add_argument("--nodes", type=int, default=5, help="Number of honest nodes")
    parser.add_argument("--sybil", type=int, default=1, help="Number of Sybil nodes")
    parser.add_argument("--runtime", type=int, default=5, help="Simulation runtime in steps")
    parser.add_argument("--walks", type=int, default=10000, help="Number of random walk steps")
    args = parser.parse_args()

    network = RandomWalkDetection(num_nodes=args.nodes, num_sybil_nodes=args.sybil, seed=42)
    network.run_simulation(until=args.runtime)

    start_node = next(nid for nid, node in network.nodes.items() if not node.is_sybil)
    visit_probs = network.random_walk_visits(start_node=start_node, steps=args.walks)

    print("\n--- Random Walk Visit Probabilities ---")
    for node_id, prob in sorted(visit_probs.items(), key=lambda x: x[1]):
        print(f"{node_id[:6]} - Visit Prob: {prob:.5f} (Sybil: {network.nodes[node_id].is_sybil})")

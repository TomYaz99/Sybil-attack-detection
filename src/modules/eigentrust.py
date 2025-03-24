import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.modules.trust_score import TrustScoreNetwork, TrustNode
import numpy as np
import networkx as nx

class EigenTrustNetwork(TrustScoreNetwork):
    '''
    EigenTrustNetwork builds upon the TrustScoreNetwork to calculate
    global trust scores using the EigenTrust algorithm.

    Global trust captures transitive trust across the entire network:
    - Each node's global score is recursively influenced by how much it is trusted by others,
      especially by nodes that themselves are highly trusted.

    Key insight:
        - Node A trusted by 10 untrusted nodes ≠ Node B trusted by 1 highly trusted node
        - Node B gets more global trust in EigenTrust

    The result is a probability vector (sums to 1), so with N nodes:
    - Average global trust per node ≈ 1/N
    - Significantly lower values may indicate Sybil or untrusted behavior
    '''
    def compute_global_trust_scores(self, damping_factor: float = 0.85, convergence_threshold: float = 1e-6, visualize: bool = True) -> dict:
        
        
        node_ids = list(self.nodes.keys())
        num_nodes = len(node_ids)

        # Build the row-normalized trust matrix: trust_matrix[i][j] = trust(i → j)
        # Also build a directed graph to track trust edges
        trust_matrix = np.zeros((num_nodes, num_nodes))
        trust_graph = nx.DiGraph()

        # Normalize each row to make it a probability distribution (∑_j trust(i→j) = 1)
        for i, source_node_id in enumerate(node_ids):
            trust_sum = 0.0
            for j, target_node_id in enumerate(node_ids):
                if target_node_id == source_node_id:
                    continue
                trust_score = self.nodes[source_node_id].get_reputation_score(target_node_id)
                if trust_score > 0:
                    trust_matrix[i][j] = trust_score
                    trust_sum += trust_score
                    trust_graph.add_edge(source_node_id, target_node_id, weight=trust_score)
            if trust_sum > 0:
                trust_matrix[i] /= trust_sum  # Normalize row

        # Power iteration initialization
        # Start with a uniform trust vector: everyone is equally trusted at first
        uniform_vector = np.ones(num_nodes) / num_nodes
        trust_vector = uniform_vector.copy()
        change = float('inf')
        trust_vector_history = [trust_vector.copy()]

        #Using main Eigen Trust formula
        #t_new = (1 - λ) * e + λ * Cᵀ * t
        #   -Damping factor (λ) controls how much the base trust vector e (uniform trust) influences the final score
        #       -This is what also us to take into account a nodes own trust score into calculations
        #   -Cᵀ propgates trust recursively
        #   -Stop computing when trust vector converges
        while change > convergence_threshold:
            updated_trust_vector = (1 - damping_factor) * uniform_vector + damping_factor * trust_matrix.T.dot(trust_vector)
            change = np.linalg.norm(updated_trust_vector - trust_vector)
            trust_vector = updated_trust_vector
            trust_vector_history.append(trust_vector.copy())

        return {node_ids[i]: trust_vector[i] for i in range(num_nodes)}


if __name__ == "__main__":
    
    '''
    Same type of example usage as in trust_score.py:
    python eigentrust.py --nodes 10 --sybil 3 --runtime 15
    '''
    import argparse

    parser = argparse.ArgumentParser(description="EigenTrust Reputation System")
    parser.add_argument("--nodes", type=int, default=5, help="Number of honest nodes")
    parser.add_argument("--sybil", type=int, default=1, help="Number of Sybil nodes")
    parser.add_argument("--runtime", type=int, default=5, help="Simulation runtime in steps")
    args = parser.parse_args()

    network = EigenTrustNetwork(num_nodes=args.nodes, num_sybil_nodes=args.sybil, seed=42)
    network.run_simulation(until=args.runtime)

    print("\n--- Local Trust Scores ---")
    for node_id, node in network.nodes.items():
        node.print_trust_scores(network.nodes)

    print("\n--- Global EigenTrust Scores ---")
    global_scores = network.compute_global_trust_scores()
    for node_id, score in sorted(global_scores.items(), key=lambda x: x[1], reverse=True):
        print(f"{node_id} (Sybil: {network.nodes[node_id].is_sybil}) - Global Trust: {score:.4f}")
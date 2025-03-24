import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.base.mock_network import MockNetwork,NetworkNode
import random
from typing import Any, Dict
from collections import defaultdict
import argparse

class TrustNode(NetworkNode):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trust_scores: Dict[str, Dict[str, float]] = defaultdict(lambda: {"alpha": 1.0, "beta": 1.0})


    def update_trust_score(self, other_node_id: str, action: str) -> None:
        """
        Update the alpha and beta values based on the interaction outcome.
        :param action: The action taken during the interaction ("good_interaction" or "bad_interaction").
        """
        score = self.trust_scores[other_node_id]
        if action == "good_interaction":
            score["alpha"] += 1
        else:
            score["beta"] += 1

    def get_reputation_score(self, other_node_id: str) -> float:
        """
        Calculate the reputation score based on the current alpha and beta values.
        :return: The reputation score as a float.
        """
        score = self.trust_scores[other_node_id]
        alpha, beta = score["alpha"], score["beta"]
        return alpha / (alpha + beta)
    
    def print_trust_scores(self, network_nodes: Dict[str, NetworkNode]) -> None:
        print(f"\n{self.node_id} (Sybil: {self.is_sybil})")
        for neighbor_id in self.get_neighbors():
            neighbor = network_nodes[neighbor_id]
            score = self.get_reputation_score(neighbor_id)
            print(f"  → Trust in {neighbor_id} (Sybil: {neighbor.is_sybil}): {score:.4f}")

class TrustScoreNetwork(MockNetwork):
    """
    A P2P network simulator that incorporates trust scoring based on interactions.
    """
    
    def _create_nodes(self, num_nodes: int, num_sybil_nodes: int) -> None:
        
        # Create honest nodes
        for _ in range(num_nodes):
            node = TrustNode(env=self.env, is_sybil=False)
            self.nodes[node.node_id] = node
        
        # Create Sybil nodes    
        for _ in range(num_sybil_nodes):
            node = TrustNode(env=self.env, is_sybil=True)
            self.nodes[node.node_id] = node
    
    def _basic_interaction(self, node_a_id: str, node_b_id: str) -> None:
        """
        Override the interaction method to implement good and bad interaction logic.
        """
        node_a = self.nodes[node_a_id]
        node_b = self.nodes[node_b_id]

        # Determine interaction outcome based on node type
        if node_a.is_sybil:
            # Sybil node has a higher chance of bad interaction
            if random.random() < 0.60:  #Chance of bad interaction if sybil
                action_a = "bad_interaction"
            else:
                action_a = "good_interaction"
        else:
            if random.random() < 0.2:  #Chance of bad interaction if non-sybil
                action_a = "bad_interaction"
            else:
                action_a = "good_interaction"
            
        if node_b.is_sybil:
            if random.random() < 0.60:  #Chance of bad interaction if sybil
                action_b = "bad_interaction"
            else:
                action_b = "good_interaction"
        else:
            if random.random() < 0.2:  #Chance of bad interaction if non-sybil
                action_b = "bad_interaction"
            else:
                action_b = "good_interaction"

        # Update trust scores based on the interaction
        node_a.update_trust_score(node_b_id, action_b)
        node_b.update_trust_score(node_a_id, action_a)

        # Record the interactions
        node_a.record_interaction(partner_id=node_b_id, action=action_a)
        node_b.record_interaction(partner_id=node_a_id, action=action_b)

if __name__ == "__main__":
    '''
    Ex how to run with code with diff arguments. 
    python trust_score.py --nodes 10 --sybil 3 --runtime 15
    
    Done this way for ease of testing.
    '''
    parser = argparse.ArgumentParser(description="Trust Score Simulation")
    parser.add_argument("--nodes", type=int, default=5, help="Number of honest nodes")
    parser.add_argument("--sybil", type=int, default=1, help="Number of Sybil nodes")
    parser.add_argument("--runtime", type=int, default=5, help="Simulation runtime in steps")
    args = parser.parse_args()
     
    try:
        network = TrustScoreNetwork(num_nodes=args.nodes, num_sybil_nodes=args.sybil, seed=42)
        network.run_simulation(until=args.runtime)

        print("Reputation scores by node:")
        for node_id, node in network.nodes.items():
            node.print_trust_scores(network.nodes)

        print("\nSimulation completed successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")
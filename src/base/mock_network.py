"""
mock_network.py

A foundational base class for simulating a P2P network. This version is deliberately
generic
"""

import simpy
import uuid
import random
from typing import List, Dict, Any, Tuple, Optional

class NetworkNode:
    """ 
    A generic network node that can be used to simulate a P2P network. 
    """
    def __init__( 
            self,
            node_id: Optional[str]  = None,
            is_sybil: bool = False,
            env: Optional[simpy.Environment] = None, 
        ) -> None:
        """
        Initializes a network node.

        :param node_id: A unique identifier for this node. If None, generates a UUID.
        :param is_sybil: If True, indicates this node is malicious/Sybil.
        :param env: SimPy environment for scheduling events and actions.
        """
        self.node_id = node_id or str(uuid.uuid4())
        self.is_sybil = is_sybil
        self.env = env

        self.reputation_score: float = 0.0
        self.neighbors: List[str] = []
        self.interaction_history: List[Dict[str,Any]] = []

    def __repr__(self) -> str:
        return f'Node({self.nodeid}, sybil={self.is_sybil})'
    
    def record_interaction(self, partner_id: str, action: str) -> None:
        """"
        Records an interaction with another node. Extend or modify this as you see fit to add
        more information about the interaction. if needed
        """
        self.interaction_history.append({
            'partner_id': partner_id,
            'action': action,
            'action': action
        })

    def add_neighbor(self, neighbor_id: str) -> None:
        """
        Adds a neighbor to this node's list of neighbors.
        """
        self.neighbors.append(neighbor_id)

    def remove_neighbor(self, neighbor_id: str) -> None:
        """
        Removes a neighbor from this node's list of neighbors.
        """
        self.neighbors.remove(neighbor_id)
    
    def get_neighbors(self) -> List[str]:
        """
        Returns a list of this node's neighbors.
        """
        return self.neighbors
    
    def get_interaction_history(self) -> List[Dict[str,Any]]:
        """
        Returns a list of this node's interactions.
        """
        return self.interaction_history   
    
    def get_reputation_score(self) -> float:
        """
        Returns this node's reputation score.
        """
        return self.reputation_score
    
    def set_reputation_score(self, score: float) -> None:
        """
        Sets this node's reputation score.
        """
        self.reputation_score = score

class MockNetwork:
    """
    A generic P2P network simulator.
    """

    def __init__(
            self,
            num_nodes: int = 10,
            num_sybil_nodes: int = 2,
            seed: Optional[int] = None

    ) -> None:
        """
        Seed for deterministic randomization. If None, randomness is not fixed.
        """
         # Initialize random seed for reproducibility
        if seed is not None:
            random.seed(seed)

        # SimPy environment for scheduling and running the simulation
        self.env = simpy.Environment()

        # Dictionary to store nodes by their ID
        self.nodes: Dict[str,NetworkNode] = {} 

        # Store the counts for dynamic updates
        self._num_nodes = num_nodes
        self._num_sybil_nodes = num_sybil_nodes

        # Create nodes
        self._create_nodes(num_nodes=num_nodes, num_sybil_nodes=num_sybil_nodes)

        # Establish a (random) initial topology
        self._connect_nodes_randomly()

        # Register the main simulation process with SimPy
        self.env.process(self._network_main_loop())

    def _create_nodes(self,num_nodes: int, num_sybil_nodes: int) -> None:
        """
        Internal helper to create and register both honest and Sybil nodes.
        """
         # Create honest nodes
        for _ in range(num_nodes):
            node = NetworkNode(env=self.env, is_sybil=False)
            self.nodes[node.node_id] = node

        # Create Sybil nodes
        for _ in range(num_sybil_nodes):
            sybil_node = NetworkNode(env=self.env, is_sybil=True)
            self.nodes[sybil_node.node_id] = sybil_node 

    def _connect_nodes_randomly(self,avg_degree: int = 2) -> None:
        """
        Simple example method that randomly links nodes together.
        Adjust or override as needed

        :param avg_degree: Approx. number of neighbors each node will attempt to have.
        """
        all_ids = list(self.nodes.keys())
        count = len(all_ids)

        for node_id in all_ids:
            node = self.nodes[node_id]
            potential_neighbors = random.sample(all_ids, k=min(avg_degree, count))
            # Exclude the node itself from neighbors
            potential_neighbors = [nid for nid in potential_neighbors if nid != node_id]
            # Merge with existing neighbors to avoid overwriting
            unique_neighbors = set(node.neighbors) | set(potential_neighbors)
            node.neighbors = list(unique_neighbors)

        # Ensure undirected connections
        for node_id, node in self.nodes.items():
            for neighbor_id in node.neighbors:
                neighbor_node = self.nodes[neighbor_id]
                if node_id not in neighbor_node.neighbors:
                    neighbor_node.neighbors.append(node_id)
        

    def _network_main_loop(self):
        """
        The primary simulation loop that SimPy calls as a generator process.
        Override or extend to schedule your own events and behaviors.
        """
        while True:
            # Example: each "time step" could trigger interactions among nodes
            self._simulate_interactions()

            # Wait 1 time unit before repeating
            yield self.env.timeout(1)

    def _simulate_interactions(self) -> None:
        """
        A placeholder for how nodes might interact at each step.
        This method is intentionally minimal—override/extend to incorporate:
            • Trust scoring updates
            • Random walks
            • LLM-based stuff
        """
        node_ids = list(self.nodes.keys())
        random.shuffle(node_ids)

        # Pair them up in consecutive pairs
        for i in range(0, len(node_ids), 2):
            if i+1 < len(node_ids):
                self._basic_interaction(node_ids[i], node_ids[i+1])

    def _basic_interaction(self, node_a_id: str, node_b_id: str) -> None:
        """
        Example method of an interaction between two nodes.
        Extend or override to inject trust logic, logs, or other stuff.
        """
        node_a = self.nodes[node_a_id]
        node_b = self.nodes[node_b_id]

        # Minimal logging of a message send/receive
        node_a.record_interaction(partner_id=node_b_id, action="send_msg")
        node_b.record_interaction(partner_id=node_a_id, action="receive_msg")

    def run_simulation(self, until: int = 10) -> None:
        start_time = self.env.now
        end_time = start_time + until
        print(f"Running simulation from time {start_time} to time {end_time}...")
        self.env.run(until=end_time)
        print("Simulation completed.")


    def add_node(self, is_sybil: bool = False) -> str:
        """
        add a new node to the network at runtime.

        :param is_sybil: Whether the newly added node is malicious (Sybil).
        :return: The node_id of the newly created node.
        """
        new_node = NetworkNode(env=self.env, is_sybil=is_sybil)
        self.nodes[new_node.node_id] = new_node
        return new_node.node_id

    def remove_node(self, node_id: str) -> None:
        """
        Remove a node from the network. Also removes it from neighbors' lists.

        :param node_id: The ID of the node to remove.
        :raises KeyError: If node_id is not found in the network.
        """
        if node_id not in self.nodes:
            raise KeyError(f"Node {node_id} does not exist.")

        # Remove references from other nodes
        for n_id, node in self.nodes.items():
            if node_id in node.neighbors:
                node.neighbors.remove(node_id)

        # Remove the node
        del self.nodes[node_id]

    def gather_network_metrics(self) -> Dict[str, Any]:
        """
        Collect network info.
        to include domain-specific stats (e.g., average trust, random-walk results, etc.).

        :return: Dictionary of basic metrics about the current network state.
        """
        total_nodes = len(self.nodes)
        sybil_nodes = sum(1 for n in self.nodes.values() if n.is_sybil)
        honest_nodes = total_nodes - sybil_nodes
        degrees = [len(node.neighbors) for node in self.nodes.values()]
        avg_degree = sum(degrees)/total_nodes if total_nodes else 0.0

        return {
            "total_nodes": total_nodes,
            "sybil_nodes": sybil_nodes,
            "honest_nodes": honest_nodes,
            "avg_degree": avg_degree,
        }

    def get_node(self, node_id: str) -> NetworkNode:
        """
        Retrieve a node by its ID.

        :param node_id: The unique ID of the node.
        :return: The node instance.
        :raises KeyError: If node_id is not found in the network.
        """
        if node_id not in self.nodes:
            raise KeyError(f"Node {node_id} not found in the network.")
        return self.nodes[node_id]

    def snapshot(self) -> None:
        """
        Simple method to print current state of the network for debugging.
        """
        print("--- Network Snapshot ---")
        for node_id, node in self.nodes.items():
            print(
                f"Node ID: {node_id}, "
                f"Sybil: {node.is_sybil}, "
                f"Neighbors: {len(node.neighbors)}"
            )
        print("------------------------")

    @property
    def num_nodes(self) -> int:
        """Returns the current number of non-Sybil nodes."""
        return self._num_nodes

    @num_nodes.setter
    def num_nodes(self, new_value: int) -> None:
        """
        Dynamically updates the number of non-Sybil nodes.
        This ensures that the network can be resized at runtime.

        :param new_value: New number of honest nodes.
        """
        if new_value < 0:
            raise ValueError("Number of nodes cannot be negative.")
        self._num_nodes = new_value
        self._create_nodes(num_nodes=self.num_nodes, num_sybil_nodes=self.num_sybil_nodes)
        self._connect_nodes_randomly()

    @property
    def num_sybil_nodes(self) -> int:
        """Returns the current number of Sybil (malicious) nodes."""
        return self._num_sybil_nodes

    @num_sybil_nodes.setter
    def num_sybil_nodes(self, new_value: int) -> None:
        """
        Dynamically updates the number of Sybil nodes.
        Ensures that the network correctly resizes when the count changes.

        :param new_value: New number of Sybil nodes.
        """
        if new_value < 0:
            raise ValueError("Number of Sybil nodes cannot be negative.")
        self._num_sybil_nodes = new_value
        self._create_nodes(num_nodes=self.num_nodes, num_sybil_nodes=self.num_sybil_nodes)
        self._connect_nodes_randomly()

    


# Example usage:
if __name__ == "__main__":
    network = MockNetwork(num_nodes=5, num_sybil_nodes=1, seed=42)
    network.run_simulation(until=5)

    print("Initial Metrics:", network.gather_network_metrics())

    # Dynamically change node count
    network.num_nodes = 10
    network.num_sybil_nodes = 3
    print("Updated Metrics:", network.gather_network_metrics())

    network.run_simulation(until=5)
    network.snapshot()

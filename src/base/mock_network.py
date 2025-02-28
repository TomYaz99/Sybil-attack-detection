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

        
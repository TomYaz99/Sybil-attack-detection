#!/usr/bin/env python3
import sys, pandas as pd, networkx as nx, matplotlib.pyplot as plt
csv, out = sys.argv[1:3]
df = pd.read_csv(csv)
# build simple undirected graph for layout
nodes = df['node'].unique()
G = nx.Graph(); G.add_nodes_from(nodes)
pos = nx.spring_layout(G, seed=42)
scores = df.groupby('node')['score'].max().to_dict()
colors = [scores[n] for n in G]
nx.draw_networkx_nodes(G, pos, node_size=40, node_color=colors, cmap='coolwarm')
nx.draw_networkx_edges(G, pos, alpha=.2)
plt.axis('off'); plt.tight_layout(); plt.savefig(out, dpi=600)

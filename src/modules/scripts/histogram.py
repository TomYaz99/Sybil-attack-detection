#!/usr/bin/env python3
import sys, pandas as pd, matplotlib.pyplot as plt
csv, out = sys.argv[1:3]
df = pd.read_csv(csv)
hon = df[df['is_sybil']==False]['score']; syb = df[df['is_sybil']]['score']
plt.hist([hon,syb], bins=30, density=True, label=['Honest','Sybil'], alpha=.7)
plt.xlabel('Final Bayesian trust score'); plt.ylabel('Density')
plt.legend(); plt.tight_layout(); plt.savefig(out, dpi=300)


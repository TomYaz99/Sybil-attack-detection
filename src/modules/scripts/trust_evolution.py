#!/usr/bin/env python3
import sys, pandas as pd, matplotlib.pyplot as plt
csv, out = sys.argv[1:3]
df   = pd.read_csv(csv)
df_h = df[df['is_sybil']==False]
df_s = df[df['is_sybil']==True]
pivot = df.pivot(index='t', columns='node', values='score')
hon  = pivot[df_h['node'].unique()].mean(axis=1)
syb  = pivot[df_s['node'].unique()].mean(axis=1)
plt.plot(hon, label='Honest', lw=2)
plt.plot(syb, label='Sybil', lw=2)
plt.xlabel('Simulation step'); plt.ylabel('Mean trust score')
plt.title('Trust trajectory (honest vs. Sybil)')
plt.legend(); plt.tight_layout(); plt.savefig(out, dpi=300)

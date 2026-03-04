"""
Phase 6: Counterfactual Scenarios — Indonesia MSCI Downgrade DSGE Model

Three scenarios:
  A: Baseline (reform failure) — 300 bps shock, rho_phi=0.80, standard Taylor rule
  B: Reform success — 200 bps shock, rho_phi=0.50
  C: Paralysed BI (CPI targeting proxy) — same shock, but rho_i=0.95 (BI frozen)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import csv, sys, os

sys.path.insert(0, '/home/claude')
from indonesia_dsge import (
    get_baseline_params, compute_composites, build_matrices,
    solve_blanchard_kahn, simulate_irf
)

def run_scenario(name, overrides):
    print("\n" + "=" * 70)
    print(f"SCENARIO {name}")
    print("=" * 70)
    params = get_baseline_params()
    for k, v in overrides.items():
        params[k] = v
    params = compute_composites(params)
    A, B, C = build_matrices(params)
    sol = solve_blanchard_kahn(A, B, verbose=False)
    if not sol['bk_satisfied']:
        print("  BK FAILED")
        return params, None
    irf = simulate_irf(sol, params, T=40, verbose=False)
    pk_x = min(irf['x'])*100
    cu_x = sum(irf['x'][:21])*100
    pk_q = max(irf['q'])*100
    pk_cpi = max(irf['pi_cpi'])*400
    mn_i = min(irf['i_level'])*400
    print(f"  eps_phi={params['eps_phi']*10000:.0f}bps  rho_phi={params['rho_phi']}  rho_i={params['rho_i']}  phi_pi={params['phi_pi']}")
    print(f"  Peak output loss:       {pk_x:.2f}%")
    print(f"  Cumul output loss (5y): {cu_x:.2f} %-quarters")
    print(f"  Peak RER depreciation:  {pk_q:.2f}%")
    print(f"  Peak CPI inflation:     {pk_cpi:.2f}% annualised")
    print(f"  Min policy rate:        {mn_i:.2f}% annualised")
    return params, irf

print("=" * 70)
print("PHASE 6: COUNTERFACTUAL SCENARIOS")
print("=" * 70)

# Scenario A: Baseline
pa, ia = run_scenario("A: BASELINE (Reform Failure)", {})

# Scenario B: Reform success
pb, ib = run_scenario("B: REFORM SUCCESS", {'eps_phi': 0.02, 'rho_phi': 0.50})

# Scenario C: Paralysed BI
# With rho_i = 0.95, BI barely adjusts the policy rate. This approximates
# what would happen if BI targeted CPI rather than domestic inflation:
# the 44% CPI spike would prevent any easing, freezing the policy rate
# near steady state while the economy contracts.
pc, ic = run_scenario("C: PARALYSED BI (CPI Targeting Proxy)", {'rho_i': 0.95})

# Comparison
print("\n" + "=" * 70)
print("COMPARISON TABLE")
print("=" * 70)
fmt = "{:<40s} {:>9s} {:>9s} {:>9s} {:>9s} {:>9s}"
print(fmt.format("Scenario", "Peak x%", "Cumul", "Peak q%", "CPI%", "Min i%"))
print("-" * 82)
for nm, irf in [("A: Baseline (reform failure)", ia),
                ("B: Reform success", ib),
                ("C: Paralysed BI (CPI proxy)", ic)]:
    if irf is None:
        print(f"{nm:<40s}  *** FAILED ***")
        continue
    print(fmt.format(nm,
        f"{min(irf['x'])*100:.2f}",
        f"{sum(irf['x'][:21])*100:.2f}",
        f"{max(irf['q'])*100:.2f}",
        f"{max(irf['pi_cpi'])*400:.2f}",
        f"{min(irf['i_level'])*400:.2f}"))

ca = sum(ia['x'][:21])*100
cb = sum(ib['x'][:21])*100
cc = sum(ic['x'][:21])*100
print(f"\n--- Key comparisons ---")
print(f"  Reform dividend (B vs A): {abs(ca-cb):.2f} %-quarters saved over 5 years")
print(f"  Cost of paralysis (C vs A): {abs(cc-ca):.2f} %-quarters additional loss")
print(f"  Total gap (C vs B): {abs(cc-cb):.2f} %-quarters")

# ---- Plot ----
T_plot = 20
fig, axes = plt.subplots(3, 2, figsize=(13, 11))
fig.suptitle('Indonesia MSCI Downgrade \u2014 Counterfactual Scenarios',
             fontsize=14, fontweight='bold', y=0.98)
zero_kw = dict(linewidth=0.8, color='grey', linestyle='--', alpha=0.5)

all_irfs   = [ia, ib, ic]
all_labels = ['A: Baseline (reform failure)', 'B: Reform success', 'C: Paralysed BI (CPI proxy)']
all_colors = ['#1f4e79', '#2e8b57', '#c0392b']

panels = [
    (0, 0, 'x',       'Output Gap',                '% of potential GDP',             100),
    (0, 1, 'pi_H',    'Domestic Inflation (pi_H)',  '% annualised',                   400),
    (1, 0, 'pi_cpi',  'CPI Inflation (pi)',         '% annualised',                   400),
    (1, 1, 'q',       'Real Exchange Rate (q)',     '% deviation (up=depreciation)',  100),
    (2, 0, 'i_level', 'Policy Rate (i)',            '% annualised',                   400),
    (2, 1, 'phi',     'Risk Premium (phi)',         'Basis points',                   10000),
]

for row, col, var, title, ylabel, scale in panels:
    ax = axes[row, col]
    for irf, label, color in zip(all_irfs, all_labels, all_colors):
        t = irf['time'][:T_plot+1]
        y = irf[var][:T_plot+1] * scale
        ax.plot(t, y, linewidth=2.0, color=color, label=label)
    ax.axhline(0, **zero_kw)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_xlim(0, T_plot)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(4))
    ax.grid(True, alpha=0.25)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

axes[2,0].set_xlabel('Quarters after shock')
axes[2,1].set_xlabel('Quarters after shock')
handles, ll = axes[0,0].get_legend_handles_labels()
fig.legend(handles, ll, loc='lower center', ncol=3, fontsize=10,
           frameon=True, bbox_to_anchor=(0.5, 0.01))
plt.tight_layout(rect=[0, 0.05, 1, 0.94])
fig.savefig('/mnt/user-data/outputs/indonesia_dsge_counterfactuals.png',
            dpi=150, bbox_inches='tight')
print("\nChart saved to: /mnt/user-data/outputs/indonesia_dsge_counterfactuals.png")
plt.close(fig)

# ---- CSV ----
csv_path = '/mnt/user-data/outputs/indonesia_dsge_counterfactual_data.csv'
with open(csv_path, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['quarter','scenario','output_gap_pct','dom_infl_ann_pct',
                'cpi_infl_ann_pct','rer_pct','policy_rate_ann_pct','risk_premium_bps'])
    for lbl, irf in [('A_baseline',ia),('B_reform',ib),('C_paralysed',ic)]:
        for t in range(len(irf['time'])):
            w.writerow([irf['time'][t], lbl,
                f"{irf['x'][t]*100:.4f}", f"{irf['pi_H'][t]*400:.4f}",
                f"{irf['pi_cpi'][t]*400:.4f}", f"{irf['q'][t]*100:.4f}",
                f"{irf['i_level'][t]*400:.4f}", f"{irf['phi'][t]*10000:.2f}"])
print(f"CSV saved to: {csv_path}")
print("\nPhase 6 complete.")

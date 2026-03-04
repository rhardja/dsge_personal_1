"""
================================================================================
Phase 4: Blanchard-Kahn DSGE Solver for Indonesia MSCI Downgrade
================================================================================

This script implements the Small Open Economy New Keynesian (SOE-NK) model
developed in Phases 1-3 of the Indonesia MSCI downgrade project.

MODEL SUMMARY
-------------
Four log-linearised equations in four endogenous variables:
    x_t       : output gap
    pi_H_t    : domestic (producer) inflation
    q_t       : real exchange rate (increase = depreciation)
    i_tilde_t : policy rate deviation from steady state

Driven by one exogenous AR(1) shock:
    phi_t     : risk premium (MSCI-related capital outflow shock)

State-space form (from Phase 2, Section 3):
    A * E_t[z_{t+1}] = B * z_t + C * eps_t

where z_t = (x_t, pi_H_t, q_t, i_tilde_{t-1}, phi_t)'.

SOLUTION METHOD
---------------
Blanchard-Kahn (1980) eigenvalue decomposition:
1. Compute D = A^{-1} B
2. Eigendecompose D, verify BK conditions (3 unstable, 2 stable eigenvalues)
3. Partition eigenvector matrix to extract:
   - Policy functions (jump vars as functions of states)
   - State transition law (law of motion for predetermined vars)
4. Simulate impulse response functions (IRFs)

REFERENCES
----------
- Galí, J. & Monacelli, T. (2005). "Monetary Policy and Exchange Rate
  Volatility in a Small Open Economy." Review of Economic Studies 72(3).
- Blanchard, O.J. & Kahn, C.M. (1980). "The Solution of Linear Difference
  Models under Rational Expectations." Econometrica 48(5).
- Phase 0-3 documents of this project for model derivation and calibration.

Author: Claude (Anthropic) for Ray's Indonesia DSGE project
Date: March 2026
================================================================================
"""

import numpy as np
from numpy.linalg import inv, eig, solve
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import warnings


# ==============================================================================
# SECTION 1: STRUCTURAL PARAMETERS (from Phase 3 Calibration)
# ==============================================================================

def get_baseline_params():
    """
    Return dictionary of baseline structural parameters.
    
    These are the 13 parameters calibrated in Phase 3, grouped by category.
    Each value is accompanied by a comment explaining its economic meaning
    and the reasoning behind the choice.
    
    Returns
    -------
    dict
        Dictionary with parameter names as keys and float values.
    """
    params = {}
    
    # ---- Group 1: Household preferences ----
    # beta: Discount factor. Implies a steady-state real interest rate of
    #        rho = -ln(beta) ≈ 0.01 per quarter (4% annualised).
    #        Standard value used in virtually all quarterly DSGE models.
    params['beta'] = 0.99
    
    # sigma: Inverse of the intertemporal elasticity of substitution (IES).
    #         sigma=1 is log utility (IES=1). This is the Galí-Monacelli
    #         benchmark. Higher sigma means less willingness to substitute
    #         consumption across time.
    params['sigma'] = 1.0
    
    # varphi: Inverse Frisch elasticity of labour supply.
    #          varphi=3 means a Frisch elasticity of 1/3 ≈ 0.33.
    #          A 1% temporary wage increase raises labour supply by 0.33%.
    #          This is in the mid-range of micro estimates (Chetty et al. 2011).
    params['varphi'] = 3.0
    
    # ---- Group 2: Open economy structure ----
    # alpha: Degree of openness = import share of consumption basket.
    #         Indonesia's trade-to-GDP ratio is ~42.6% (2024).
    #         alpha=0.40 is the Galí-Monacelli illustrative value and matches
    #         Indonesia's openness well.
    params['alpha'] = 0.40
    
    # eta: Elasticity of substitution between domestic and foreign goods
    #       (Armington elasticity). eta>1 means domestic and foreign goods
    #       are substitutes. This is the KEY open-economy parameter:
    #       it controls how strongly exchange rate depreciation redirects
    #       demand from imports to domestic goods (expenditure switching).
    #       We depart from GM's eta=1 to eta=1.5, consistent with the
    #       trade literature (Anderson & van Wincoop 2004).
    params['eta'] = 1.5
    
    # gamma: Elasticity of substitution across different foreign goods.
    #         We keep the GM benchmark of 1. This is a second-order parameter
    #         that only enters through omega; hard to identify separately.
    params['gamma'] = 1.0
    
    # ---- Group 3: Nominal rigidity ----
    # theta: Calvo price stickiness parameter. Each quarter, fraction (1-theta)
    #         of firms can re-optimise their price. theta=0.75 means 25% reset
    #         per quarter → average price duration = 4 quarters (1 year).
    #         Standard in the literature; conservative for Indonesia (EM prices
    #         likely adjust faster).
    params['theta'] = 0.75
    
    # ---- Group 4: Monetary policy (Taylor rule) ----
    # rho_i: Interest rate smoothing. BI moves rates gradually — 75% of last
    #          period's rate persists. Estimated range for BI: 0.7-0.9.
    #          We use the lower end to be conservative about amplification.
    params['rho_i'] = 0.75
    
    # phi_pi: Taylor rule inflation response coefficient. For every 1pp of
    #          inflation above target, BI raises the nominal rate by 1.5pp.
    #          Satisfies the Taylor principle (phi_pi > 1): the real rate rises
    #          by 0.5pp, which is contractionary and stabilises inflation.
    params['phi_pi'] = 1.5
    
    # phi_y: Taylor rule output gap response. For every 1pp negative output
    #          gap, BI lowers the nominal rate by 0.5pp. Standard Taylor (1993).
    params['phi_y'] = 0.5
    
    # ---- Group 5: Financial accelerator and shock process ----
    # psi: Financial accelerator sensitivity. The term -psi*phi_t in the IS
    #       curve captures credit tightening beyond the exchange rate channel
    #       (bank lending contraction, equity collapse, balance sheet effects).
    #       psi=0.15 delivers ~2x amplification relative to pure ER channel.
    #       Set psi=0 to isolate the exchange rate mechanism only.
    params['psi'] = 0.15
    
    # rho_phi: Persistence of the risk premium shock. AR(1) coefficient.
    #           phi_t = rho_phi * phi_{t-1} + eps_t.
    #           rho_phi=0.80 → half-life ≈ 3.1 quarters (~9 months).
    #           After 8 quarters (2 years), shock has decayed to ~17%.
    params['rho_phi'] = 0.80
    
    # eps_phi: Size of the one-time risk premium innovation (in levels).
    #           0.03 = 300 basis points. Derived from Goldman Sachs $10B
    #           central outflow scenario scaled by historical episode benchmarks.
    params['eps_phi'] = 0.03
    
    return params


# ==============================================================================
# SECTION 2: COMPOSITE PARAMETER COMPUTATION
# ==============================================================================

def compute_composites(params):
    """
    Compute the composite (reduced-form) parameters from deep structural ones.
    
    The chain is (Phase 2, Section 5):
        (alpha, sigma, eta, gamma) → omega → sigma_alpha
        (beta, theta) → lambda_calvo
        (sigma_alpha, varphi, lambda_calvo) → kappa_alpha
    
    Parameters
    ----------
    params : dict
        Structural parameters from get_baseline_params().
    
    Returns
    -------
    dict
        Updated dictionary including composite parameters.
    """
    p = params.copy()
    
    # Unpack for readability
    sigma = p['sigma']
    alpha = p['alpha']
    eta   = p['eta']
    gamma = p['gamma']
    beta  = p['beta']
    theta = p['theta']
    varphi = p['varphi']
    
    # ---- Step 1: omega ----
    # omega aggregates the three substitution elasticities and openness.
    # omega = sigma*gamma + (1-alpha)*(sigma*eta - 1)
    # When sigma=eta=gamma=1 (GM special case), omega=1 and the open economy
    # IS curve is isomorphic to the closed economy.
    # With our calibration (eta=1.5), omega=1.3 > 1, so openness amplifies
    # the effect of interest rate changes on demand.
    p['omega'] = sigma * gamma + (1 - alpha) * (sigma * eta - 1)
    
    # ---- Step 2: sigma_alpha ----
    # The "effective" intertemporal elasticity in the open economy.
    # sigma_alpha = sigma / [(1-alpha) + alpha*omega]
    # When omega > 1, sigma_alpha < sigma: the economy is MORE responsive
    # to interest rate changes (exchange rate movements amplify demand).
    p['sigma_alpha'] = sigma / ((1 - alpha) + alpha * p['omega'])
    
    # ---- Step 3: lambda (Calvo composite) ----
    # lambda = (1 - beta*theta)*(1 - theta) / theta
    # This is purely a function of the discount factor and price stickiness.
    # Higher theta (stickier prices) → smaller lambda → flatter Phillips curve.
    p['lambda_calvo'] = (1 - beta * theta) * (1 - theta) / theta
    
    # ---- Step 4: kappa_alpha (Phillips curve slope) ----
    # kappa_alpha = lambda * (sigma_alpha + varphi)
    # Combines the Calvo friction (lambda) with marginal cost sensitivity
    # to the output gap (sigma_alpha + varphi).
    #   - sigma_alpha: consumption/wealth channel (higher output → higher
    #     consumption → higher marginal rate of substitution → higher wages)
    #   - varphi: labour supply channel (higher output → more employment →
    #     higher marginal disutility of work → higher wages)
    p['kappa_alpha'] = p['lambda_calvo'] * (p['sigma_alpha'] + varphi)
    
    # ---- Steady-state real rate ----
    # rho = -ln(beta) ≈ 0.01 per quarter (4% annualised)
    # This is the intercept in the Taylor rule and the foreign real rate r*.
    p['rho'] = -np.log(beta)
    
    return p


# ==============================================================================
# SECTION 3: MATRIX CONSTRUCTION (A, B, C)
# ==============================================================================

def build_matrices(p):
    """
    Construct the A, B, C matrices for the state-space system:
    
        A * E_t[z_{t+1}] = B * z_t + C * eps_t
    
    State vector: z_t = (x_t, pi_H_t, q_t, i_tilde_{t-1}, phi_t)'
    
    The ordering follows Phase 2, Section 3:
        Row 1: IS curve (with Taylor rule substituted)
        Row 2: NKPC
        Row 3: UIP (with Taylor rule substituted)
        Row 4: Taylor rule identity (advancing i_tilde by one period)
        Row 5: Shock process (AR(1) for phi)
    
    Key feature: A is lower-triangular, so A^{-1} is trivial to compute.
    This was achieved by substituting the Taylor rule into the IS and UIP
    equations, eliminating i_t as an independent forward-looking variable.
    
    Parameters
    ----------
    p : dict
        Parameters including composites (output of compute_composites).
    
    Returns
    -------
    A : ndarray (5,5)
        Coefficient matrix on E_t[z_{t+1}].
    B : ndarray (5,5)
        Coefficient matrix on z_t.
    C : ndarray (5,1)
        Shock loading vector.
    """
    # Unpack parameters used in the matrices
    sa     = p['sigma_alpha']  # sigma_alpha (effective IES in open economy)
    beta   = p['beta']
    ka     = p['kappa_alpha']  # Phillips curve slope
    rho_i  = p['rho_i']       # interest rate smoothing
    phi_pi = p['phi_pi']      # Taylor rule inflation response
    phi_y  = p['phi_y']       # Taylor rule output gap response
    psi    = p['psi']         # financial accelerator
    rho_phi = p['rho_phi']    # shock persistence
    
    # Pre-compute frequently used combinations
    inv_sa = 1.0 / sa                       # 1/sigma_alpha ≈ 1.12
    one_m_rho = 1.0 - rho_i                 # (1 - rho_i) = 0.25
    tr_y   = one_m_rho * phi_y / sa         # Taylor rule output term in IS: (1-rho_i)*phi_y/sigma_alpha
    tr_pi  = one_m_rho * phi_pi / sa        # Taylor rule inflation term in IS: (1-rho_i)*phi_pi/sigma_alpha
    
    # =========================================================================
    # A MATRIX (5x5) — coefficients on E_t[z_{t+1}]
    # =========================================================================
    # This matrix is LOWER-TRIANGULAR by construction (Phase 2, Section 3).
    # The off-diagonal elements come from:
    #   A[0,1] = 1/sigma_alpha  (from E_t[pi_{H,t+1}] in the IS curve)
    #   A[2,1] = -1             (from -E_t[pi_{H,t+1}] in the UIP equation)
    #
    # Diagonal: [1, beta, 1, 1, 1]
    # =========================================================================
    A = np.array([
        [1.0,    inv_sa,  0.0,  0.0,  0.0],   # Row 1: IS curve
        [0.0,    beta,    0.0,  0.0,  0.0],   # Row 2: NKPC
        [0.0,   -1.0,     1.0,  0.0,  0.0],   # Row 3: UIP
        [0.0,    0.0,     0.0,  1.0,  0.0],   # Row 4: Taylor identity
        [0.0,    0.0,     0.0,  0.0,  1.0],   # Row 5: Shock process
    ])
    
    # =========================================================================
    # B MATRIX (5x5) — coefficients on z_t
    # =========================================================================
    # After substituting the Taylor rule into IS and UIP, the B matrix contains
    # all time-t terms. The structure reflects the three channels:
    #
    #   Column 1 (x_t):        output gap affects IS (directly + Taylor rule)
    #                           and NKPC (marginal cost)
    #   Column 2 (pi_H_t):     inflation affects IS (Taylor rule) and UIP
    #                           (Taylor rule)
    #   Column 3 (q_t):        real ER affects only UIP (current level)
    #   Column 4 (i_tilde_{t-1}): lagged policy rate (smoothing)
    #   Column 5 (phi_t):      risk premium enters IS (financial accelerator)
    #                           and UIP (directly, with opposite sign)
    #
    # Key entries explained:
    #   B[0,0] = 1 + (1-rho_i)*phi_y/sigma_alpha
    #          = 1 + Taylor rule output term. The "1" comes from x_t on the
    #            RHS of the IS curve; the rest from substituting the Taylor rule.
    #   B[0,4] = psi: financial accelerator — direct demand contraction
    #   B[2,4] = -1: risk premium enters UIP with negative sign (higher phi
    #            depreciates q, but in the rearranged equation q_t appears on
    #            the RHS with +1, so phi_t enters as -1 on the RHS too)
    # =========================================================================
    B = np.array([
        # Columns:  x_t,                    pi_H_t,               q_t,   i_tilde_{t-1},  phi_t
        [1.0 + tr_y * sa / inv_sa,  tr_pi * sa / inv_sa,   0.0,   rho_i * inv_sa,     psi       ],   # Row 1: IS
        [-ka,                        1.0,                   0.0,   0.0,                 0.0       ],   # Row 2: NKPC
        [one_m_rho * phi_y,          one_m_rho * phi_pi,    1.0,   rho_i,              -1.0       ],   # Row 3: UIP
        [one_m_rho * phi_y,          one_m_rho * phi_pi,    0.0,   rho_i,               0.0       ],   # Row 4: Taylor identity
        [0.0,                        0.0,                   0.0,   0.0,                 rho_phi   ],   # Row 5: Shock
    ])
    
    # Wait — let me re-derive B[0,:] more carefully.
    # The IS row after Taylor substitution (Phase 2 eq at line 265):
    #   E_t[x_{t+1}] + (1/sa)*E_t[pi_{H,t+1}] = 
    #       [1 + (1-rho_i)*phi_y/sa] * x_t 
    #       + [(1-rho_i)*phi_pi/sa] * pi_H_t
    #       + [rho_i/sa] * i_tilde_{t-1}
    #       + psi * phi_t
    #
    # So B[0,:] should be:
    B[0, 0] = 1.0 + one_m_rho * phi_y / sa      # coefficient on x_t
    B[0, 1] = one_m_rho * phi_pi / sa            # coefficient on pi_H_t
    B[0, 2] = 0.0                                 # q_t does not appear in IS
    B[0, 3] = rho_i / sa                          # coefficient on i_tilde_{t-1}
    B[0, 4] = psi                                  # financial accelerator
    
    # =========================================================================
    # C VECTOR (5x1) — shock loading
    # =========================================================================
    # The innovation eps_t^phi enters ONLY through the shock equation (Row 5):
    #   phi_{t+1} = rho_phi * phi_t + eps_{t+1}
    # So C = (0, 0, 0, 0, 1)'.
    # =========================================================================
    C = np.array([[0.0], [0.0], [0.0], [0.0], [1.0]])
    
    return A, B, C


# ==============================================================================
# SECTION 4: BLANCHARD-KAHN SOLUTION
# ==============================================================================

def solve_blanchard_kahn(A, B, n_forward=3, n_predetermined=2, verbose=True):
    """
    Solve the linear rational expectations system using Blanchard-Kahn (1980).
    
    System: A * E_t[z_{t+1}] = B * z_t
    
    where z_t = (forward_vars, predetermined_vars)'.
    
    ALGORITHM OVERVIEW
    ------------------
    1. Form D = A^{-1} * B (the "companion matrix").
    2. Eigendecompose D = P * Lambda * P^{-1}, where Lambda is diagonal
       with eigenvalues and P is the matrix of right eigenvectors.
    3. BK condition: number of eigenvalues with |lambda| > 1 must equal
       the number of forward-looking (jump) variables. If fewer → multiple
       equilibria (indeterminacy). If more → no stable solution.
    4. Sort eigenvalues: stable (|lambda| < 1) first, unstable (|lambda| > 1)
       last.
    5. Partition the INVERSE eigenvector matrix Q = P^{-1} into blocks
       corresponding to stable/unstable eigenvalues and forward/predetermined
       variables.
    6. The policy function (expressing jump variables as functions of
       predetermined states) is:
           forward_t = -Q_uf^{-1} * Q_up * predetermined_t
       where Q_uf is the unstable-forward block and Q_up is the
       unstable-predetermined block.
    7. The state transition law for predetermined variables is:
           predetermined_{t+1} = M * predetermined_t
       where M is derived from the stable eigenvalue block.
    
    Parameters
    ----------
    A : ndarray (n,n)
        LHS coefficient matrix.
    B : ndarray (n,n)
        RHS coefficient matrix.
    n_forward : int
        Number of forward-looking (jump) variables. Default: 3 (x, pi_H, q).
    n_predetermined : int
        Number of predetermined variables. Default: 2 (i_tilde_{t-1}, phi_t).
    verbose : bool
        If True, print diagnostic information about eigenvalues.
    
    Returns
    -------
    dict with keys:
        'D'           : the companion matrix A^{-1}B
        'eigenvalues' : eigenvalues of D
        'P'           : right eigenvector matrix (sorted)
        'Lambda'      : sorted eigenvalues (diagonal)
        'policy_matrix' : F such that forward_t = F @ predetermined_t
        'state_transition' : M for predetermined_{t+1} = M @ predetermined_t
        'bk_satisfied' : bool, whether BK conditions hold
    """
    n = A.shape[0]
    assert n == n_forward + n_predetermined, \
        f"Dimension mismatch: n={n}, n_forward={n_forward}, n_pre={n_predetermined}"
    
    # ------------------------------------------------------------------
    # Step 1: Compute D = A^{-1} * B
    # ------------------------------------------------------------------
    # Since A is lower-triangular (by construction from Phase 2), we could
    # use forward substitution. But for clarity and generality, we use
    # numpy's solve (which detects triangular structure internally).
    D = solve(A, B)
    
    if verbose:
        print("=" * 70)
        print("BLANCHARD-KAHN SOLUTION DIAGNOSTICS")
        print("=" * 70)
        print(f"\nSystem dimension: {n}")
        print(f"Forward-looking variables: {n_forward} (x, pi_H, q)")
        print(f"Predetermined variables:   {n_predetermined} (i_tilde_{{t-1}}, phi)")
        print(f"\nCondition number of A: {np.linalg.cond(A):.4f}")
        print(f"Determinant of A:      {np.linalg.det(A):.6f}")
    
    # ------------------------------------------------------------------
    # Step 2: Eigendecompose D
    # ------------------------------------------------------------------
    eigenvalues, P = eig(D)
    
    # Compute moduli (absolute values) for sorting
    moduli = np.abs(eigenvalues)
    
    if verbose:
        print(f"\nEigenvalues of D = A^{{-1}}B:")
        print(f"{'Eigenvalue':>25s}  {'|λ|':>10s}  {'Type':>10s}")
        print("-" * 50)
        for i, (ev, mod) in enumerate(zip(eigenvalues, moduli)):
            ev_type = "STABLE" if mod < 1 else "UNSTABLE"
            # Format complex eigenvalues nicely
            if np.isreal(ev):
                ev_str = f"{ev.real:>12.6f}"
            else:
                ev_str = f"{ev.real:>8.4f} + {ev.imag:>8.4f}i"
            print(f"  λ_{i+1} = {ev_str}  {mod:>10.6f}  {ev_type:>10s}")
    
    # ------------------------------------------------------------------
    # Step 3: Check BK condition
    # ------------------------------------------------------------------
    # IMPORTANT: Eigenvalues with |lambda| >= 1 are treated as UNSTABLE.
    # The BK method requires that forward-looking variables have their
    # eigenspace components set to zero, which applies to eigenvalues
    # ON or OUTSIDE the unit circle. An eigenvalue at exactly 1.0
    # (unit root) is non-stationary and must be counted as unstable.
    #
    # We use a small tolerance (1 - 1e-10) to handle numerical precision.
    # ------------------------------------------------------------------
    stability_threshold = 1.0 - 1e-10
    n_unstable = np.sum(moduli > stability_threshold)
    n_stable = np.sum(moduli <= stability_threshold)
    bk_satisfied = (n_unstable == n_forward)
    
    if verbose:
        print(f"\nBK condition check:")
        print(f"  Unstable eigenvalues (|λ|>1): {n_unstable}")
        print(f"  Required (= n_forward):       {n_forward}")
        if bk_satisfied:
            print(f"  ✓ BK condition SATISFIED — unique stable solution exists.")
        else:
            if n_unstable < n_forward:
                print(f"  ✗ INDETERMINACY — too few unstable eigenvalues.")
            else:
                print(f"  ✗ NO STABLE SOLUTION — too many unstable eigenvalues.")
    
    if not bk_satisfied:
        warnings.warn("Blanchard-Kahn conditions NOT satisfied!")
        return {
            'D': D, 'eigenvalues': eigenvalues, 'P': P,
            'bk_satisfied': False
        }
    
    # ------------------------------------------------------------------
    # Step 4: Sort eigenvalues — stable first, unstable last
    # ------------------------------------------------------------------
    # Stable: |lambda| < 1 (strictly inside unit circle)
    # Unstable: |lambda| >= 1 (on or outside unit circle)
    # We sort so that the n_predetermined smallest-modulus eigenvalues
    # come first (stable), and the n_forward largest come last (unstable).
    sort_idx = np.argsort(moduli)  # ascending order of |lambda|
    eigenvalues_sorted = eigenvalues[sort_idx]
    P_sorted = P[:, sort_idx]      # reorder columns of eigenvector matrix
    
    if verbose:
        print(f"\nSorted eigenvalues (stable first):")
        for i, ev in enumerate(eigenvalues_sorted):
            marker = "stable" if i < n_predetermined else "unstable"
            print(f"  λ_{i+1} = {ev.real:>10.6f}  (|λ| = {abs(ev):.6f}, {marker})")
    
    # ------------------------------------------------------------------
    # Step 5: Partition Q = P^{-1}
    # ------------------------------------------------------------------
    # The state vector is ordered as: z_t = (forward | predetermined)
    # But OUR ordering is: z_t = (x, pi_H, q | i_tilde_{t-1}, phi_t)
    # So forward vars are indices [0,1,2] and predetermined are [3,4].
    #
    # Q = P^{-1} transforms from original coordinates to eigenvalue space.
    # We partition Q into blocks:
    #
    #   Q = [ Q_sf  Q_sp ]   where s = stable (first n_pre rows),
    #       [ Q_uf  Q_up ]         u = unstable (last n_fwd rows),
    #                              f = forward (first n_fwd cols),
    #                              p = predetermined (last n_pre cols).
    #
    # The policy function is:
    #   forward_t = -Q_uf^{-1} @ Q_up @ predetermined_t
    #
    # The state transition is:
    #   predetermined_{t+1} = (Lambda_s + Q_sp @ F) @ Q_sp^{-1} ... 
    #   but more directly from the stable block.
    # ------------------------------------------------------------------
    Q = inv(P_sorted)
    
    # Partition indices
    f_idx = list(range(n_forward))                    # [0, 1, 2] — forward vars
    p_idx = list(range(n_forward, n))                 # [3, 4]    — predetermined vars
    s_idx = list(range(n_predetermined))              # [0, 1]    — stable eigenvalues
    u_idx = list(range(n_predetermined, n))           # [2, 3, 4] — unstable eigenvalues
    
    # Extract the four blocks of Q
    Q_sf = Q[np.ix_(s_idx, f_idx)]   # stable rows × forward cols
    Q_sp = Q[np.ix_(s_idx, p_idx)]   # stable rows × predetermined cols
    Q_uf = Q[np.ix_(u_idx, f_idx)]   # unstable rows × forward cols
    Q_up = Q[np.ix_(u_idx, p_idx)]   # unstable rows × predetermined cols
    
    # Stable eigenvalues as diagonal matrix
    Lambda_s = np.diag(eigenvalues_sorted[s_idx])
    
    # ------------------------------------------------------------------
    # Step 6: Policy function
    # ------------------------------------------------------------------
    # For the solution to be non-explosive, the component of the state
    # vector in the unstable eigenspace must be zero for all t ≥ 0.
    # This gives the RESTRICTION:
    #   Q_uf @ forward_t + Q_up @ predetermined_t = 0
    # 
    # Solving for the forward (jump) variables:
    #   forward_t = -Q_uf^{-1} @ Q_up @ predetermined_t
    #             =  F @ predetermined_t
    #
    # F is the (n_forward × n_predetermined) POLICY MATRIX.
    # It tells us how the jump variables respond on impact to the states.
    # ------------------------------------------------------------------
    F = -solve(Q_uf, Q_up)
    
    # Take only the real part (imaginary parts should be negligible
    # if eigenvalues come in conjugate pairs, which they do here)
    F = F.real
    
    if verbose:
        print(f"\nPolicy matrix F ({n_forward}×{n_predetermined}):")
        print(f"  (forward_t = F @ predetermined_t)")
        labels_fwd = ['x_t', 'pi_H_t', 'q_t']
        labels_pre = ['i_tilde_{t-1}', 'phi_t']
        print(f"{'':>12s}", end="")
        for lp in labels_pre:
            print(f"{lp:>18s}", end="")
        print()
        for i, lf in enumerate(labels_fwd):
            print(f"  {lf:>10s}", end="")
            for j in range(n_predetermined):
                print(f"{F[i, j]:>18.6f}", end="")
            print()
    
    # ------------------------------------------------------------------
    # Step 7: State transition law
    # ------------------------------------------------------------------
    # From the stable block of the eigendecomposition:
    #   s_{t+1} = Lambda_s @ s_t
    # where s_t = Q_sf @ forward_t + Q_sp @ predetermined_t
    #           = Q_sp @ predetermined_t + Q_sf @ F @ predetermined_t
    #           = (Q_sp + Q_sf @ F) @ predetermined_t
    #
    # Let G = Q_sp + Q_sf @ F. Then s_t = G @ predetermined_t, and:
    #   G @ predetermined_{t+1} = Lambda_s @ G @ predetermined_t
    #   predetermined_{t+1} = G^{-1} @ Lambda_s @ G @ predetermined_t
    #                       = M @ predetermined_t
    # ------------------------------------------------------------------
    G = Q_sp + Q_sf @ F
    M = solve(G, Lambda_s @ G)
    M = M.real
    
    if verbose:
        print(f"\nState transition matrix M ({n_predetermined}×{n_predetermined}):")
        print(f"  (predetermined_{{t+1}} = M @ predetermined_t)")
        for i, lp in enumerate(labels_pre):
            print(f"  {lp:>18s}", end="")
            for j in range(n_predetermined):
                print(f"{M[i, j]:>18.6f}", end="")
            print()
        
        # Verify: eigenvalues of M should equal the stable eigenvalues of D
        M_eigs = np.linalg.eigvals(M)
        print(f"\n  Eigenvalues of M: {[f'{e.real:.6f}' for e in M_eigs]}")
        print(f"  (Should match stable eigenvalues of D)")
    
    return {
        'D': D,
        'eigenvalues': eigenvalues_sorted,
        'P': P_sorted,
        'Q': Q,
        'F': F,                     # policy matrix
        'M': M,                     # state transition
        'bk_satisfied': True,
    }


# ==============================================================================
# SECTION 5: IMPULSE RESPONSE FUNCTION SIMULATION
# ==============================================================================

def simulate_irf(solution, params, T=40, verbose=True):
    """
    Simulate impulse response functions to a one-time risk premium shock.
    
    The experiment:
        - At t=0, the risk premium innovation eps_0^phi = eps_phi (300 bps).
        - For t >= 1, eps_t^phi = 0 (no further shocks).
        - The shock decays as phi_t = rho_phi^t * eps_phi.
        - All variables start at zero (steady state) before the shock.
    
    Using the solved policy function (F) and state transition (M):
        1. Set initial predetermined state: s_0 = (0, eps_phi)'
           (i_tilde_{-1} = 0, phi_0 = eps_phi)
        2. Compute forward vars: forward_0 = F @ s_0
        3. Advance states: s_1 = M @ s_0
        4. Repeat for T periods.
    
    Then compute auxiliary variables (Phase 2, Section 4):
        - Terms of trade:       s_t = q_t / (1 - alpha)
        - CPI inflation:        pi_t = pi_H_t + alpha/(1-alpha) * (q_t - q_{t-1})
        - Nominal depreciation: Delta_e_t = (q_t - q_{t-1}) + pi_t
        - Policy rate level:    i_t = i_tilde_t + rho
        - Real interest rate:   r_t = i_t - E_t[pi_H_{t+1}]
    
    Parameters
    ----------
    solution : dict
        Output from solve_blanchard_kahn().
    params : dict
        Full parameter dictionary (with composites).
    T : int
        Number of quarters to simulate. Default: 40 (10 years).
    verbose : bool
        If True, print selected IRF values.
    
    Returns
    -------
    dict
        Time series for all endogenous and auxiliary variables.
        Each value is a 1D numpy array of length T+1 (including t=0).
    """
    F = solution['F']      # (3 × 2) policy matrix
    M = solution['M']      # (2 × 2) state transition
    
    eps_phi  = params['eps_phi']
    alpha    = params['alpha']
    rho      = params['rho']
    rho_i    = params['rho_i']
    phi_pi   = params['phi_pi']
    phi_y    = params['phi_y']
    
    # Allocate storage for all variables (T+1 periods: t = 0, 1, ..., T)
    # Endogenous (core):
    x      = np.zeros(T + 1)   # output gap
    pi_H   = np.zeros(T + 1)   # domestic inflation
    q      = np.zeros(T + 1)   # real exchange rate
    i_til  = np.zeros(T + 1)   # policy rate deviation (i_tilde_t)
    phi    = np.zeros(T + 1)   # risk premium
    
    # Auxiliary:
    s_tot  = np.zeros(T + 1)   # terms of trade
    pi_cpi = np.zeros(T + 1)   # CPI inflation
    delta_e = np.zeros(T + 1)  # nominal depreciation
    i_level = np.zeros(T + 1)  # policy rate level (= i_tilde + rho)
    r_real = np.zeros(T + 1)   # real interest rate
    
    # ------------------------------------------------------------------
    # Initial conditions (t = 0)
    # ------------------------------------------------------------------
    # Before the shock: all variables at zero (steady state).
    # The shock hits phi at t=0: phi_0 = eps_phi.
    # The lagged policy rate i_tilde_{-1} = 0 (was at steady state).
    #
    # So the initial predetermined state vector is:
    #   predetermined_0 = (i_tilde_{-1}, phi_0) = (0, eps_phi)
    # ------------------------------------------------------------------
    predetermined = np.array([0.0, eps_phi])
    
    for t in range(T + 1):
        # -- Extract predetermined variables --
        i_til_lagged = predetermined[0]   # i_tilde_{t-1}
        phi_t        = predetermined[1]   # phi_t
        
        # -- Compute forward (jump) variables via policy function --
        # forward_t = F @ predetermined_t
        forward = F @ predetermined
        x_t    = forward[0]   # output gap
        pi_H_t = forward[1]   # domestic inflation
        q_t    = forward[2]   # real exchange rate
        
        # -- Compute current policy rate from Taylor rule --
        # i_tilde_t = rho_i * i_tilde_{t-1} + (1-rho_i)*(phi_pi*pi_H_t + phi_y*x_t)
        i_til_t = rho_i * i_til_lagged + (1 - rho_i) * (phi_pi * pi_H_t + phi_y * x_t)
        
        # -- Store core variables --
        x[t]     = x_t
        pi_H[t]  = pi_H_t
        q[t]     = q_t
        i_til[t] = i_til_t
        phi[t]   = phi_t
        
        # -- Compute auxiliary variables --
        # Terms of trade: s_t = q_t / (1 - alpha)
        s_tot[t] = q_t / (1 - alpha)
        
        # CPI inflation: pi_t = pi_H_t + alpha/(1-alpha) * Delta(q_t)
        if t == 0:
            delta_q = q_t - 0.0       # q_{-1} = 0 (steady state)
        else:
            delta_q = q_t - q[t - 1]
        pi_cpi[t] = pi_H_t + alpha / (1 - alpha) * delta_q
        
        # Nominal depreciation: Delta_e_t = Delta(q_t) + pi_t - pi*
        # (pi* = 0 in steady state)
        delta_e[t] = delta_q + pi_cpi[t]
        
        # Policy rate level: i_t = i_tilde_t + rho
        i_level[t] = i_til_t + rho
        
        # Real interest rate: r_t = i_t - E_t[pi_{H,t+1}]
        # We need E_t[pi_{H,t+1}]. From the policy function:
        #   pi_H_{t+1} = F[1,:] @ predetermined_{t+1}
        # And predetermined_{t+1} = M @ predetermined_t.
        if t < T:
            predetermined_next = M @ predetermined
            forward_next = F @ predetermined_next
            pi_H_next = forward_next[1]
        else:
            # At the last period, use the NKPC to back out:
            pi_H_next = 0.0  # approximately zero far out
        r_real[t] = i_level[t] - pi_H_next
        
        # -- Advance state for next period --
        # predetermined_{t+1} = M @ predetermined_t
        if t < T:
            predetermined = M @ predetermined
    
    # ------------------------------------------------------------------
    # Print selected IRF values
    # ------------------------------------------------------------------
    if verbose:
        print("\n" + "=" * 70)
        print("IMPULSE RESPONSE FUNCTIONS — Selected Quarters")
        print("=" * 70)
        print(f"\nShock: Risk premium innovation = {eps_phi*100:.0f} bps at t=0")
        print(f"{'Quarter':>8s} {'Output':>10s} {'Dom.Infl':>10s} {'CPI Infl':>10s} "
              f"{'RER (q)':>10s} {'Pol.Rate':>10s} {'Risk Pr.':>10s}")
        print(f"{'':>8s} {'Gap %':>10s} {'% q-o-q':>10s} {'% q-o-q':>10s} "
              f"{'% dev':>10s} {'% ann':>10s} {'bps':>10s}")
        print("-" * 70)
        
        # Print quarters 0, 1, 2, 3, 4, 8, 12, 16, 20
        report_quarters = [0, 1, 2, 3, 4, 8, 12, 16, 20]
        for t in report_quarters:
            if t <= T:
                print(f"{t:>8d} {x[t]*100:>10.3f} {pi_H[t]*100:>10.3f} "
                      f"{pi_cpi[t]*100:>10.3f} {q[t]*100:>10.3f} "
                      f"{i_level[t]*400:>10.2f} {phi[t]*10000:>10.1f}")
    
    return {
        # Core variables (in decimal: 0.01 = 1%)
        'x': x,              # output gap
        'pi_H': pi_H,        # domestic inflation (quarterly)
        'q': q,              # real exchange rate
        'i_tilde': i_til,    # policy rate deviation from SS
        'phi': phi,           # risk premium
        
        # Auxiliary variables
        's': s_tot,           # terms of trade
        'pi_cpi': pi_cpi,    # CPI inflation (quarterly)
        'delta_e': delta_e,  # nominal depreciation (quarterly)
        'i_level': i_level,  # policy rate level
        'r_real': r_real,    # real interest rate
        
        # Time vector
        'T': T,
        'time': np.arange(T + 1),
    }


# ==============================================================================
# SECTION 6: PLOTTING
# ==============================================================================

def plot_irfs(irf, params, save_path=None):
    """
    Generate a publication-quality 3×2 panel of impulse response functions.
    
    Panel layout:
        (1,1) Output Gap             (1,2) Domestic Inflation
        (2,1) CPI Inflation          (2,2) Real Exchange Rate
        (3,1) Policy Rate            (3,2) Risk Premium
    
    All variables are shown in economically interpretable units:
        - Output gap: percent of potential GDP
        - Inflation: annualised percentage points
        - Real exchange rate: percent deviation from SS (+ = depreciation)
        - Policy rate: annualised percent
        - Risk premium: basis points
    
    Parameters
    ----------
    irf : dict
        Output from simulate_irf().
    params : dict
        Parameter dictionary (used for annotation).
    save_path : str or None
        If provided, save figure to this path.
    """
    t = irf['time']
    
    # Limit plot to 20 quarters (5 years) for clarity
    T_plot = min(20, irf['T'])
    t_plot = t[:T_plot + 1]
    
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    fig.suptitle(
        'Indonesia MSCI Downgrade — Impulse Response Functions\n'
        f'Risk premium shock: {params["eps_phi"]*10000:.0f} bps, '
        f'persistence: ρ_φ = {params["rho_phi"]:.2f}, '
        f'financial accelerator: ψ = {params["psi"]:.2f}',
        fontsize=13, fontweight='bold', y=0.98
    )
    
    # Common styling
    line_kwargs = dict(linewidth=2.0, color='#1f4e79')
    zero_kwargs = dict(linewidth=0.8, color='grey', linestyle='--', alpha=0.6)
    
    # --- Panel (1,1): Output Gap ---
    ax = axes[0, 0]
    ax.plot(t_plot, irf['x'][:T_plot+1] * 100, **line_kwargs)
    ax.axhline(0, **zero_kwargs)
    ax.set_title('Output Gap', fontsize=11, fontweight='bold')
    ax.set_ylabel('% of potential GDP')
    
    # --- Panel (1,2): Domestic Inflation ---
    ax = axes[0, 1]
    ax.plot(t_plot, irf['pi_H'][:T_plot+1] * 400, **line_kwargs)
    ax.axhline(0, **zero_kwargs)
    ax.set_title('Domestic Inflation (π_H)', fontsize=11, fontweight='bold')
    ax.set_ylabel('% annualised')
    
    # --- Panel (2,1): CPI Inflation ---
    ax = axes[1, 0]
    ax.plot(t_plot, irf['pi_cpi'][:T_plot+1] * 400, **line_kwargs)
    ax.axhline(0, **zero_kwargs)
    ax.set_title('CPI Inflation (π)', fontsize=11, fontweight='bold')
    ax.set_ylabel('% annualised')
    
    # --- Panel (2,2): Real Exchange Rate ---
    ax = axes[1, 1]
    ax.plot(t_plot, irf['q'][:T_plot+1] * 100, **line_kwargs)
    ax.axhline(0, **zero_kwargs)
    ax.set_title('Real Exchange Rate (q)', fontsize=11, fontweight='bold')
    ax.set_ylabel('% deviation (↑ = depreciation)')
    
    # --- Panel (3,1): Policy Rate ---
    ax = axes[2, 0]
    ax.plot(t_plot, irf['i_level'][:T_plot+1] * 400, **line_kwargs)
    ax.axhline(params['rho'] * 400, **zero_kwargs)
    ax.set_title('Policy Rate (i)', fontsize=11, fontweight='bold')
    ax.set_ylabel('% annualised')
    ax.set_xlabel('Quarters after shock')
    
    # --- Panel (3,2): Risk Premium ---
    ax = axes[2, 1]
    ax.plot(t_plot, irf['phi'][:T_plot+1] * 10000, **line_kwargs)
    ax.axhline(0, **zero_kwargs)
    ax.set_title('Risk Premium (φ)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Basis points')
    ax.set_xlabel('Quarters after shock')
    
    # Common formatting for all panels
    for ax in axes.flat:
        ax.set_xlim(0, T_plot)
        ax.xaxis.set_major_locator(mticker.MultipleLocator(4))
        ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nFigure saved to: {save_path}")
    
    plt.close(fig)
    return fig


# ==============================================================================
# SECTION 7: DETERMINACY CHECK
# ==============================================================================

def check_determinacy(params):
    """
    Verify the Taylor principle for our calibration.
    
    The condition (Bullard & Mitra 2002, adapted for open economy):
        kappa_alpha * (phi_pi - 1) + (1 - beta) * phi_y > 0
    
    This ensures the central bank raises the real interest rate in response
    to inflationary pressure, preventing self-fulfilling inflation spirals.
    
    Parameters
    ----------
    params : dict
        Parameters with composites.
    
    Returns
    -------
    bool
        True if determinacy condition is satisfied.
    """
    ka      = params['kappa_alpha']
    phi_pi  = params['phi_pi']
    phi_y   = params['phi_y']
    beta    = params['beta']
    
    # Taylor principle criterion
    criterion = ka * (phi_pi - 1) + (1 - beta) * phi_y
    
    print(f"\nDeterminacy check (Taylor principle):")
    print(f"  κ_α × (φ_π - 1) + (1-β) × φ_y = "
          f"{ka:.4f} × {phi_pi - 1:.1f} + {1-beta:.4f} × {phi_y:.1f} = {criterion:.4f}")
    print(f"  Required: > 0")
    print(f"  {'✓ SATISFIED' if criterion > 0 else '✗ VIOLATED'}")
    
    return criterion > 0


# ==============================================================================
# SECTION 8: MAIN EXECUTION
# ==============================================================================

def main():
    """
    Main execution: calibrate, solve, simulate, and plot.
    
    This function orchestrates the complete workflow:
        1. Load baseline parameters
        2. Compute composite parameters
        3. Check determinacy (Taylor principle)
        4. Build A, B, C matrices
        5. Solve via Blanchard-Kahn
        6. Simulate impulse response functions
        7. Generate and save plots
    """
    print("=" * 70)
    print("INDONESIA MSCI DOWNGRADE — SOE-NK DSGE MODEL")
    print("Phase 4: Blanchard-Kahn Implementation")
    print("=" * 70)
    
    # ------------------------------------------------------------------
    # Step 1: Load and compute parameters
    # ------------------------------------------------------------------
    print("\n--- Step 1: Parameters ---")
    params = get_baseline_params()
    params = compute_composites(params)
    
    print(f"\nStructural parameters:")
    print(f"  β={params['beta']}, σ={params['sigma']}, φ={params['varphi']}")
    print(f"  α={params['alpha']}, η={params['eta']}, γ={params['gamma']}")
    print(f"  θ={params['theta']}")
    print(f"  ρ_i={params['rho_i']}, φ_π={params['phi_pi']}, φ_y={params['phi_y']}")
    print(f"  ψ={params['psi']}, ρ_φ={params['rho_phi']}, ε_φ={params['eps_phi']}")
    
    print(f"\nComposite parameters:")
    print(f"  ω     = {params['omega']:.4f}")
    print(f"  σ_α   = {params['sigma_alpha']:.4f}")
    print(f"  λ     = {params['lambda_calvo']:.4f}")
    print(f"  κ_α   = {params['kappa_alpha']:.4f}")
    print(f"  ρ (SS real rate) = {params['rho']:.4f} ({params['rho']*400:.2f}% ann.)")
    
    # ------------------------------------------------------------------
    # Step 2: Determinacy check
    # ------------------------------------------------------------------
    print("\n--- Step 2: Determinacy ---")
    is_determinate = check_determinacy(params)
    if not is_determinate:
        print("ERROR: Model is indeterminate. Check Taylor rule parameters.")
        return
    
    # ------------------------------------------------------------------
    # Step 3: Build matrices
    # ------------------------------------------------------------------
    print("\n--- Step 3: Build A, B, C matrices ---")
    A, B, C = build_matrices(params)
    
    print("\nA matrix (lower-triangular, coefficients on E_t[z_{t+1}]):")
    print(np.array2string(A, precision=4, suppress_small=True))
    
    print("\nB matrix (coefficients on z_t):")
    print(np.array2string(B, precision=4, suppress_small=True))
    
    print("\nC vector (shock loading):")
    print(C.flatten())
    
    # ------------------------------------------------------------------
    # Step 4: Blanchard-Kahn solution
    # ------------------------------------------------------------------
    print("\n--- Step 4: Blanchard-Kahn Solution ---")
    solution = solve_blanchard_kahn(A, B, n_forward=3, n_predetermined=2)
    
    if not solution['bk_satisfied']:
        print("ERROR: BK conditions not satisfied. Aborting.")
        return
    
    # ------------------------------------------------------------------
    # Step 5: Simulate IRFs
    # ------------------------------------------------------------------
    print("\n--- Step 5: Impulse Response Functions ---")
    irf = simulate_irf(solution, params, T=40)
    
    # Print summary statistics
    print(f"\n--- Summary Statistics ---")
    print(f"  Peak output loss:        {min(irf['x'])*100:.2f}% of potential GDP")
    print(f"  Peak CPI inflation:      {max(irf['pi_cpi'])*400:.2f}% annualised")
    print(f"  Peak RER depreciation:   {max(irf['q'])*100:.2f}%")
    print(f"  Peak policy rate:        {max(irf['i_level'])*400:.2f}% annualised")
    print(f"  Cumulative output loss:  {sum(irf['x'][:21])*100:.2f}%-quarters (5yr)")
    
    # ------------------------------------------------------------------
    # Step 6: Generate plots
    # ------------------------------------------------------------------
    print("\n--- Step 6: Generating Plots ---")
    plot_irfs(irf, params, save_path='/mnt/user-data/outputs/indonesia_dsge_irfs.png')
    
    # ------------------------------------------------------------------
    # Also save the numerical IRF data to CSV for further analysis
    # ------------------------------------------------------------------
    print("\n--- Saving IRF data ---")
    import csv
    csv_path = '/mnt/user-data/outputs/indonesia_dsge_irf_data.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'quarter', 'output_gap_pct', 'domestic_inflation_ann_pct',
            'cpi_inflation_ann_pct', 'real_exchange_rate_pct',
            'policy_rate_ann_pct', 'real_rate_ann_pct',
            'risk_premium_bps', 'terms_of_trade_pct',
            'nominal_depreciation_ann_pct'
        ])
        for t in range(irf['T'] + 1):
            writer.writerow([
                t,
                f"{irf['x'][t]*100:.4f}",
                f"{irf['pi_H'][t]*400:.4f}",
                f"{irf['pi_cpi'][t]*400:.4f}",
                f"{irf['q'][t]*100:.4f}",
                f"{irf['i_level'][t]*400:.4f}",
                f"{irf['r_real'][t]*400:.4f}",
                f"{irf['phi'][t]*10000:.2f}",
                f"{irf['s'][t]*100:.4f}",
                f"{irf['delta_e'][t]*400:.4f}",
            ])
    print(f"  IRF data saved to: {csv_path}")
    
    print("\n" + "=" * 70)
    print("Phase 4 complete. Ready for Phase 5 (Analysis & Sensitivity).")
    print("=" * 70)
    
    return params, solution, irf


# ==============================================================================
# ENTRY POINT
# ==============================================================================

if __name__ == '__main__':
    params, solution, irf = main()

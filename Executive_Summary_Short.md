# Executive Summary (Short Version)

*Modeling Indonesia's MSCI Downgrade: A Small Open Economy DSGE Approach*

---

## The problem

In January 2026, MSCI froze all positive index changes for Indonesian securities, citing opaque ownership structures and unreliable free-float data. If Indonesia fails to reform by May 2026, it faces reclassification from Emerging Market to Frontier Market. Goldman Sachs estimates this would trigger $7.8–13.4 billion in forced outflows — a sudden stop driven not by economic fundamentals but by index mechanics.

This project asks: **how bad could the macroeconomic fallout be, and what can Indonesia do about it?**

## The model

We build a small open economy New Keynesian DSGE model based on Galí and Monacelli (2005), extended with a financial accelerator to capture the credit tightening and equity collapse that accompany sudden stops. The model has four core equations (IS curve, Phillips curve, uncovered interest parity, Taylor rule) and is driven by a single exogenous shock: a 300 basis point risk premium innovation representing the MSCI downgrade, calibrated from Goldman Sachs outflow estimates.

The model is solved using the Blanchard-Kahn (1980) eigenvalue decomposition method and simulated as impulse response functions over a 10-year horizon.

## Baseline results (reform failure)

If the downgrade proceeds, the model predicts on impact: a **17.3% real exchange rate depreciation** (consistent with historical episodes like the 2013 taper tantrum), a **0.66% output gap** (GDP falling below potential), and a **44.6% annualised CPI inflation spike** — though this is a one-quarter price-level adjustment, not persistent inflation. Bank Indonesia, targeting domestic inflation (which actually *falls* due to weak demand), would ease the policy rate from 4.0% to about 2.9%. The cumulative output loss over five years is approximately 1.2 percentage-point-quarters.

## Counterfactual scenarios

We compare three scenarios:

| | Reform success | Baseline | Paralysed BI |
|---|---|---|---|
| Peak output loss | -0.28% | -0.66% | -0.98% |
| Cumulative loss (5yr) | -0.30 %-q | -1.20 %-q | -1.39 %-q |
| Peak RER depreciation | 4.2% | 17.3% | 15.1% |

**Reform is the most powerful lever.** If Indonesia acts decisively before the deadline, cumulative output losses fall by 75%. The exchange rate barely moves and CPI inflation peaks at just 10.8% instead of 44.6%.

**Monetary policy regime matters.** If Bank Indonesia is unable to ease — as would happen under CPI targeting, where the headline inflation spike prevents rate cuts — the peak recession deepens by nearly 50%. The total gap between best-case (reform) and worst-case (paralysed BI) is roughly $15 billion in lost output.

## Key takeaway

The single most effective response is structural reform: raise the free-float minimum, improve ownership disclosure, and satisfy MSCI's accessibility requirements. No amount of monetary policy finesse can substitute for removing the shock at its source. Conditional on the downgrade proceeding, the domestic inflation targeting framework gives Bank Indonesia the space to ease into the recession without being distracted by the transitory CPI spike — a result consistent with the theoretical optimality results in Galí and Monacelli (2005).

---

*Disclaimer: This is an independent modeling exercise for educational purposes. The model has not been estimated against data or validated against historical episodes. All numerical results are illustrative of mechanism and direction, not point forecasts. See the full project disclaimer for details.*

# Executive Summary (Short Version)

*Modeling Indonesia's MSCI Downgrade: A Small Open Economy DSGE Approach*

---

## The problem

In January 2026, MSCI froze all positive index changes for Indonesian securities, citing opaque ownership structures and unreliable free-float data. If Indonesia fails to reform by May 2026, it faces reclassification from Emerging Market to Frontier Market. Goldman Sachs estimates this would trigger $7.8–13.4 billion in forced outflows — a sudden stop driven not by economic fundamentals but by index mechanics.

This project asks: **how bad could the macroeconomic fallout be, and what can Indonesia do about it?**

## What is a DSGE model?

DSGE stands for Dynamic Stochastic General Equilibrium. That is a mouthful, so let us break it down word by word.

**Dynamic** means the model tracks how the economy evolves over time — not just what happens on impact, but quarter by quarter as the shock fades and the economy adjusts. This is what produces the impulse response functions (the time-series charts) that form the core output of this project.

**Stochastic** means the model includes random shocks — unpredictable events that hit the economy and push it away from its resting point. In our case, the shock is the MSCI downgrade: a sudden, unexpected spike in the risk premium that Indonesia pays to borrow from international capital markets. The model does not predict *when* shocks happen; it tells us *what happens after* they do.

**General Equilibrium** means all markets in the model clear simultaneously. When the exchange rate moves, that affects import prices, which affects inflation, which affects Bank Indonesia's interest rate decision, which feeds back into the exchange rate. Nothing happens in isolation — every variable responds to every other variable, and the model solves for all of them at once. This is the key difference from simpler approaches (like multiplier analysis or reduced-form regressions), which typically examine one channel at a time.

In practice, a DSGE model is a system of equations — in our case, four — representing the behaviour of households, firms, the central bank, and the foreign sector. Each equation has a clear economic interpretation: households optimise consumption over time (IS curve), firms set prices subject to frictions (Phillips curve), financial markets arbitrage across currencies (UIP), and the central bank follows a policy rule (Taylor rule). The model is calibrated with numerical parameter values (price stickiness, trade openness, central bank aggressiveness, etc.) and then solved using linear algebra to produce quantitative predictions.

### What DSGE can and cannot do

**DSGE is good for:**

- Tracing the *transmission mechanism* of a shock — understanding which channels matter, how variables interact, and why the economy responds the way it does. Our model shows, for example, that the output cost comes primarily from the financial accelerator and the interest rate channel, while the exchange rate depreciation actually *helps* by redirecting demand toward domestic goods.

- Comparing *policy counterfactuals* in a disciplined way. Because the model is structural (each equation represents an economic relationship, not just a statistical correlation), we can change one piece — say, the monetary policy rule — and trace the consequences without worrying that other relationships will break. This is what Phase 6 does when it compares domestic inflation targeting to a paralysed central bank.

- Providing *internally consistent* scenarios. Every variable in every scenario respects the same economic logic. The exchange rate, inflation, output, and interest rates all move together in a way that is mutually consistent — something that is difficult to achieve with ad hoc assumptions or scenario planning in spreadsheets.

**DSGE is not good for:**

- *Point forecasting*. DSGE models are not designed to predict that GDP will be exactly 4.7% next quarter. They are tools for understanding mechanisms and comparing scenarios, not for generating precise numerical forecasts. The specific numbers in this project (17.3% depreciation, 0.66% output loss) should be read as "roughly this order of magnitude, driven by these channels" rather than "exactly this."

- *Capturing nonlinear or extreme events*. Our model is log-linearised around a steady state, which means it approximates the economy as if small deviations from normal are the only thing that happens. If the shock is large enough to trigger a banking crisis, a sovereign default, or a currency peg collapse, the linear approximation breaks down. The 17% depreciation in our baseline is at the edge of where this assumption is comfortable.

- *Modelling institutions and politics*. DSGE models have no government budget constraint, no elections, no regulatory agencies, no corporate boardrooms. Our model cannot tell you whether the Indonesian parliament will pass free-float reform, or how OJK will implement new disclosure rules, or whether Bank Indonesia's governor will face political pressure to defend the rupiah. It can only tell you what happens *after* those decisions are made.

- *Replacing judgment*. A DSGE model is a disciplined way to organise economic thinking, not a substitute for it. The parameter choices, the model specification, and the scenario design all require human judgment. The model enforces internal consistency on that judgment, but it cannot tell you whether the judgment itself is correct.

### How to use this project

Read the results as a *structured thought experiment*: given these assumptions about the Indonesian economy, this is what the economic logic implies. If you disagree with an assumption (say, you think the shock will be larger, or Bank Indonesia will be more aggressive), the model framework tells you which direction that changes the answer. Use the counterfactual comparison to understand relative magnitudes ("reform cuts losses by 75%") rather than absolute numbers ("output falls by exactly 0.66%"). And always keep the disclaimer in mind — this is a learning exercise, not a policy prescription.

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

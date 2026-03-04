# Executive Summary (Long Version)

*Modeling Indonesia's MSCI Downgrade: A Small Open Economy DSGE Approach*

---

## Background

On January 28, 2026, MSCI froze all positive index changes for Indonesian securities. The reason was not economic weakness but governance: opaque ownership structures, unreliable free-float data, and concerns over coordinated trading that distorts price formation. MSCI set a May 2026 deadline — if Indonesia fails to demonstrate sufficient progress on transparency, it faces a weighting reduction and potential reclassification from Emerging Market (EM) to Frontier Market (FM).

The announcement triggered the Jakarta Composite Index's largest two-day decline in decades, erasing approximately $80 billion in market capitalisation. Goldman Sachs estimated that a full reclassification would force $7.8 billion in passive fund outflows (rising to $13.4 billion if FTSE Russell follows suit) — a sudden stop driven entirely by index mechanics, not by any deterioration in Indonesia's underlying economic fundamentals.

This project builds a macroeconomic model to answer three questions: how large would the output and exchange rate effects be? How quickly would the economy recover? And what policy choices can mitigate the damage?

## Model framework

We use a small open economy New Keynesian DSGE model based on the canonical framework of Galí and Monacelli (2005), extended with a reduced-form financial accelerator parameter ($\psi$) that captures the credit tightening, equity collapse, and balance sheet deterioration that accompany sudden stops in emerging markets.

The model consists of four log-linearised equations:

1. **IS curve** — An open-economy Euler equation linking the output gap to expected future output, the real interest rate, and the risk premium shock (via the financial accelerator).

2. **New Keynesian Phillips curve** — Domestic (producer) inflation driven by the output gap and expected future inflation, with the slope determined by the degree of price stickiness (Calvo parameter $\theta = 0.75$, implying prices reset on average once a year).

3. **Uncovered interest parity (UIP)** — The real exchange rate adjusts to equate expected returns on domestic and foreign assets, modified by the risk premium shock. This equation generates the Dornbusch (1976) overshooting result: the exchange rate must depreciate *beyond* its long-run equilibrium on impact so that expected future appreciation compensates investors for the elevated risk premium.

4. **Taylor rule** — Bank Indonesia sets the policy rate in response to domestic inflation and the output gap, with interest rate smoothing. Crucially, the Taylor rule targets domestic (producer) inflation, not CPI — a distinction that proves decisive for the counterfactual analysis.

The model is driven by a single exogenous shock: a 300 basis point risk premium innovation ($\varepsilon_0^\phi = 0.03$) representing the MSCI downgrade, injected at quarter 0 and decaying via an AR(1) process with persistence $\rho_\phi = 0.80$ (half-life of approximately 3 quarters). The shock size is calibrated from Goldman Sachs estimates: $10 billion in outflows represents roughly 0.73% of Indonesian GDP, scaled to a risk premium using the empirical relationship from historical EM sudden-stop episodes.

## Solution method

The model is cast in state-space form ($A \, \mathbb{E}_t[\mathbf{z}_{t+1}] = B \, \mathbf{z}_t + C \, \varepsilon_t$) and solved using the Blanchard-Kahn (1980) eigenvalue decomposition. The 5×5 system has three forward-looking variables (output gap, domestic inflation, real exchange rate) and two predetermined variables (lagged policy rate, risk premium), requiring exactly three unstable eigenvalues for a unique stable solution. The solver confirmed this condition is satisfied, with eigenvalues at 0.384, 0.800 (stable) and 1.000, 1.099, 1.795 (unstable).

The solution yields a policy matrix $F$ (expressing forward-looking variables as functions of the predetermined states) and a state transition matrix $M$ (governing the evolution of predetermined variables). These two matrices are iterated forward to produce impulse response functions (IRFs) — the dynamic paths of all endogenous variables following the single shock at $t=0$.

## Baseline results

If the downgrade proceeds and reforms fail, the model produces the following impact-quarter effects:

- **Real exchange rate:** 17.3% depreciation — large but consistent with historical episodes (the rupiah depreciated approximately 25% during the 2013 taper tantrum; Pakistan's currency fell 15-20% around its 2021 MSCI reclassification).

- **Output gap:** -0.66% of potential GDP. The contraction operates through three channels: the financial accelerator (credit tightening and equity collapse), the interest rate channel (even though BI eases, rates remain above their natural level), and reduced consumption from wealth effects.

- **CPI inflation:** 44.6% annualised (11.2% in the quarter). This is a one-time price-level adjustment driven by the exchange rate depreciation feeding through to import prices, not a persistent inflationary process. CPI inflation turns negative in quarter 1 as the exchange rate begins appreciating back from its overshoot.

- **Domestic inflation:** -1.6% annualised. This is the key distinction — domestic (producer) inflation *falls* because the output gap is deeply negative, reducing firms' marginal costs and pricing pressure. The divergence between CPI and domestic inflation is what makes the monetary policy regime choice so consequential.

- **Policy rate:** Drops from 4.0% (steady state) to about 2.9% on impact. Bank Indonesia, observing falling domestic inflation and a widening output gap, provides monetary easing — which cushions the output contraction and allows the exchange rate to act as a shock absorber.

The recovery is gradual but complete. The output gap closes within about 8 quarters (2 years). The exchange rate takes slightly longer, returning to within 1% of steady state by quarter 12 (3 years). The cumulative output loss over five years is 1.2 percentage-point-quarters — meaningful but moderate, equivalent to losing roughly 0.3% of one year's GDP spread over the adjustment period.

## Counterfactual analysis

### Scenario B: Reform success

If Indonesia acts decisively — raising the free-float minimum, improving foreign ownership disclosure, and satisfying MSCI's accessibility criteria — we model this as a smaller (200 bps) and less persistent ($\rho_\phi = 0.50$) shock. Markets assign meaningful probability to reform success, compressing both the initial panic and the duration of elevated risk premia.

The results are dramatically better: peak output loss of just -0.28% (less than half the baseline), exchange rate depreciation of only 4.2% (one-quarter of the baseline), and CPI inflation peaking at 10.8% annualised. The cumulative five-year output loss falls to 0.30 %-quarters — a 75% reduction from the baseline. The risk premium is effectively gone within four quarters.

The reform dividend is large because both the size and persistence channels reinforce each other. The total risk premium exposure (area under the AR(1) curve) is approximately four times smaller under reform than under failure.

### Scenario C: Paralysed Bank Indonesia

Same shock as the baseline, but Bank Indonesia cannot ease because it is constrained by the CPI inflation spike. We model this by raising the rate smoothing parameter to $\rho_i = 0.95$ — meaning BI adjusts only 5% of the warranted change each quarter, effectively freezing the policy rate near its steady state.

This scenario approximates what would happen under CPI inflation targeting. A CPI-targeting central bank observing 44.6% annualised headline inflation would conclude that inflation is far above target and either hold rates steady or tighten. Even though the CPI spike is a transitory one-quarter price-level adjustment that is already reversing, the headline number would prevent any easing.

The consequences are significant: peak output loss deepens to -0.98% (nearly 50% worse than the baseline), and cumulative five-year loss rises to 1.39 %-quarters. The policy rate barely moves — dropping to only 3.7% instead of the baseline's 2.9% — providing almost no monetary cushion.

An interesting side effect: the paralysed-BI scenario actually produces a *smaller* exchange rate depreciation (15.1% vs 17.3%) and a *lower* CPI spike (38.4% vs 44.6%), because higher interest rates attract capital and stabilise the currency. But this "benefit" comes at a steep cost in real output — the exchange rate cannot perform its role as a shock absorber, and the economy bears the full contractionary impact.

### Comparison

| Metric | B: Reform | A: Baseline | C: Paralysed BI |
|---|---|---|---|
| Peak output loss | -0.28% | -0.66% | -0.98% |
| Cumulative output loss (5yr) | -0.30 %-q | -1.20 %-q | -1.39 %-q |
| Peak RER depreciation | 4.2% | 17.3% | 15.1% |
| Peak CPI inflation (ann.) | 10.8% | 44.6% | 38.4% |
| Minimum policy rate | 3.73% | 2.93% | 3.71% |
| Reform dividend vs A | 0.90 %-q saved | — | 0.20 %-q additional loss |

The total gap between best and worst case is 1.09 percentage-point-quarters over five years. For an economy of Indonesia's size (~$1.4 trillion GDP), this represents roughly $15 billion in lost output.

## Policy implications

**First and foremost: reform.** The model's clearest message is that structural reform before the MSCI deadline dominates all other policy responses. Reducing the shock at its source is worth far more than optimising the monetary response to a shock that has already landed. Raising the free-float minimum from 7.5% toward international norms, improving ownership transparency, and addressing the coordinated trading concerns MSCI flagged would compress both the initial market reaction and the persistence of any risk premium.

**Preserve the domestic inflation targeting framework.** Bank Indonesia's choice to target domestic (producer) inflation rather than CPI is not a technicality — it is what allows monetary policy to support the economy through the transition. A CPI-targeting regime would force BI to fight a transitory import price spike at the cost of deepening the recession. This result is consistent with the theoretical optimality findings in Galí and Monacelli (2005) and is quantitatively important in the sudden-stop context.

**Communicate clearly.** The divergence between CPI inflation (spiking to 44%) and domestic inflation (falling to -1.6%) will confuse the public and the media. BI will need to explain convincingly why it is easing policy while the rupiah is falling and consumer prices are rising. The model provides the analytical framework for that communication: the CPI spike is a one-off price-level adjustment, not an inflation process, and fighting it with rate hikes would worsen the recession without meaningfully stabilising the exchange rate.

---

*Disclaimer: This is an independent modeling exercise conducted for educational and intellectual purposes. The model has not been estimated against data, no sensitivity analysis has been performed, and the numerical results should be interpreted as illustrative of mechanism and direction, not as point forecasts. See `Disclaimer.md` for full details.*

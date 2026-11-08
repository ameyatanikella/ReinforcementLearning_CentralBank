# Can a Reinforcement Learning Agent Run Monetary Policy?

A DDPG agent is given one lever — the nominal interest rate — and dropped into a simulated economy whose dynamics were learned from real Federal Reserve data. Its objective is the central bank's: drive inflation to 2% and close the output gap.

The point of interest is that the environment is not a hand-written textbook model. The transition equations *are* neural networks fit to FRED data, so the agent is optimizing against estimated macroeconomic dynamics rather than an economist's closed-form approximation of them.

## How it is built

**1. Learn the dynamics from data.** Inflation is computed as the year-over-year percent change in the GDP deflator (FRED series `GDPDEF`), paired with output-gap data over the same period. Two `MLPRegressor` networks (one tanh hidden layer, `lbfgs` solver, 85/15 train/test split) are fit to predict next-period inflation and next-period output gap. The tanh activation follows the Deutsche Bundesbank paper the approach is based on.

**2. Freeze them into an environment.** The fitted weights and biases are read out of scikit-learn and re-implemented as an explicit forward pass (`transition()`) inside a custom `gymnasium.Env` called `Econov`, with uniform noise added so the economy is stochastic.

**3. Let an agent push on it.**

| Component | Definition |
| --- | --- |
| Action | Nominal interest rate, continuous `Box(-2, 20)` |
| Observation | `[π_t, π_{t-1}, y_t, y_{t-1}]` — two lags of each, so the problem stays Markovian despite transitions depending on history |
| Reward | `−0.5·(π − 2.0)² − 0.5·y` — a quadratic loss around the Fed's 2% target plus an output-gap penalty |
| Initial state | Q1/Q2 2025 Fed data (π ≈ 2.9–3.0), with the starting interest rate randomized so the agent cannot memorize one path |
| Episode end | Truncates once π ∈ [1.7, 2.3] and y ∈ [−0.3, 0.3]; terminates at 4000 steps |

Transitions feed the networks up to three lags of inflation and two of the interest rate, which is why the observation carries lags rather than a single snapshot.

**4. Train.** `DDPG` from stable-baselines3 with an `MlpPolicy` and normal action noise (σ = 0.5), `lr=1e-3`, buffer 100k, batch 256, γ = 0.99, τ = 0.005, for 500k timesteps. TD3 and SAC are imported as alternatives for comparison.

## Design notes

- **Equal weighting on the two gaps.** The reward uses `w = 0.5` for both the inflation gap and the output gap. An Okun's-law-derived weight would be closer to 0.125, but the literature on social welfare supports weighting them equally, so that is what is used here.
- **Bounded action and state space.** Inflation, output gap and interest rate are all clipped to plausible ranges. Without those bounds, a reward-maximizing agent will happily discover an interest rate of 10,000% and an economy that no longer resembles anything real.
- **Why DDPG.** Setting a rate is a continuous-control problem, which rules out the value-based methods that would need the rate discretized into bins.

## Running it

```bash
pip install stable_baselines3 gymnasium fredapi scikit-learn pandas matplotlib
```

Open `Centralbank.ipynb` and add a [FRED API key](https://fred.stlouisfed.org/docs/api/api_key.html) in the data-extraction cell (`Fred(api_key=...)` is a placeholder — no key is committed to this repository). Then run the cells in order: data extraction → fit the transition networks → define the environment → train.

## Status

Work in progress, and a single notebook rather than a packaged project. The economics and the architecture are worked out; the code does not yet run end to end. Known gaps, in the order they need fixing:

- `reset()` is missing its `self` parameter, and `step()` returns bare `observation`, `reward` and `steps` names instead of the `self.*` attributes it set.
- `np.random.uniform[0.0, 1.0]` uses brackets where it needs parentheses, in the three-lag fallback branch.
- `inflataion_params` is a typo for the inflation parameter dict.
- The `observation_space` upper bounds are accidentally set to the *minimum* values, so the declared box is degenerate.
- `truncated` and `terminated` are swapped relative to the Gymnasium convention — the step returns `(obs, reward, truncated, terminated, info)`.
- The output-gap initial values (13.19, 13.74) are large for a percentage output gap and are worth re-checking against the source series.

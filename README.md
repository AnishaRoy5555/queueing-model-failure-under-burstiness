# When Queueing Models Break

**Why classical queueing theory underestimates congestion, and how bursty arrivals break it.**

Standard M/M/1 and M/M/k formulas assume Poisson arrivals. Real systems have bursty arrivals: restaurant rushes, trading order flow, network packet storms. This project quantifies how large the resulting errors are when assumptions fail. In high-utilization systems, this leads to order-of-magnitude errors in wait-time estimates.

## Key Finding

| Arrival Pattern | Wait Time vs M/M/1 Prediction |
|-----------------|-------------------------------|
| Poisson (baseline) | 0% (theory matches) |
| Batch arrivals (mean size 2) | **+92%** |
| Hawkes process (self-exciting) | **+79%** |

*Same mean arrival rate. Radically different delays.*

**M/M/1 formulas underestimate wait times by up to 2× under realistic conditions.**

Multiple servers help but don't eliminate the problem:

| System | Batch Penalty | Hawkes Penalty |
|--------|---------------|----------------|
| M/M/1 (ρ = 0.8) | +92% | +79% |
| M/M/4 (ρ = 0.8) | +59% | +44% |

*Parallelism dampens burstiness. It does not remove it.*

## Why This Matters

**Trading systems**: Order flow is self-exciting (Hawkes). A limit order book queue model using Poisson assumptions will underestimate fill times, leading to mispriced latency risk.

**Infrastructure capacity planning**: Server provisioning based on M/M/k assumes smooth arrivals. Real traffic is bursty, leading to chronic underprovisioning.

**Call centers and restaurants**: Customer arrivals cluster (families, groups, lunch rushes). Erlang-C staffing models are optimistic, leading to understaffing during peaks.

## The Variance Insight

At high utilization, simulation "errors" aren't bugs. They're finite-sample effects.

At ρ = 0.95:
```
Var[L] = ρ / (1 - ρ)² = 380
```

A single long busy period can skew the entire sample mean. Reliable mean estimates at high load require 500,000+ samples. This explains why real systems "feel worse" than models predict: you're experiencing the tail, not the mean.

**Implication**: Point estimates are misleading unless confidence intervals are reported.

## Technical Details

*This section documents the mathematics and implementation for completeness. The key insights are above.*

### Derivations (from first principles)

- **M/M/1**: Balance equations → steady-state distribution → Little's Law
- **M/M/k**: State-dependent service rates → Erlang-C formula
- **M/G/1**: Pollaczek-Khinchin formula, variance multiplier effect
- **Batch arrivals (Mˣ/M/1)**: Internal delay analysis, heavy-traffic ratio (batch size X with geometric distribution)
- **Hawkes process**: Self-excitation, branching ratio, overdispersion

All derivations in [`docs/`](docs/) with step-by-step proofs.

### Simulation

Discrete-event simulation with:
- Event-driven architecture (heapq priority queue)
- Warm-up period handling for steady-state convergence
- Time-averaged integrals for queue length metrics
- Hawkes arrivals via Ogata's thinning algorithm
- Batch arrivals with geometric size distribution

### Validated Results

Baseline models are validated first to isolate the effect of burstiness.

| Test Case | Metric | Simulation | Analytical | Error |
|-----------|--------|------------|------------|-------|
| M/M/1 (ρ=0.8) | E[Wq] | 4.02 | 4.00 | 0.5% |
| M/M/3 (ρ=0.67) | E[W] | 1.45 | 1.44 | 0.7% |
| M/M/5 (ρ=0.8) | P(wait) | 0.56 | 0.55 | 1.8% |

High-ρ cases (ρ = 0.95) show larger deviations. This is expected due to variance explosion, not simulation error.

## Project Structure

```
├── queue_simulation.py      # Core simulator + bursty arrival extensions
├── docs/
│   ├── mm1_derivation.pdf   # M/M/1 from first principles
│   ├── mmk_derivation.pdf   # M/M/k with Erlang-C
│   └── queueing_derivations.pdf  # M/G/1, batch, Hawkes, assumption failures
└── results/
    └── queueing_theory_results.pdf  # Simulation validation
```

## Running the Simulation

```bash
python queue_simulation.py
```

This runs:
1. M/M/1 and M/M/k validation tests
2. Bursty arrivals comparison (Poisson vs Batch vs Hawkes)
3. Multi-server burstiness penalty analysis

## Limitations and Extensions

- Results depend on stationary burst statistics; nonstationary regimes remain future work
- Heavy-tailed service times (M/G/1 with high C²ₛ) would further amplify variance effects
- Extension to priority queues and limit order book matching is a natural next step

## Core Insight

> Models fail not because they're wrong, but because variance dominates near saturation, and real arrivals are never Poisson.

The gap between theory and reality isn't a bug. It's the central finding.

## References

- Kleinrock, L. (1975). *Queueing Systems, Volume 1: Theory*
- Hawkes, A. G. (1971). Spectra of some self-exciting and mutually exciting point processes
- Ross, S. M. (2014). *Introduction to Probability Models*

---


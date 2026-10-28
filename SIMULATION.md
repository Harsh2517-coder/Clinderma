# Simulation study: can face-zone spot patterns help predict future conditions?

**This is a simulation, not clinical evidence.** No dataset links face patterns to the conditions people later develop, so the cohort is generated from stated assumptions
(`risk_sim.py`). Results describe what a method could detect *if* an effect of a given size existed. They never show that the effect exists.

## Design
- **People:** 8 body systems; hidden imbalance per system driven by age, BMI, stress, sleep, smoking, alcohol and family history plus unobserved noise.
  Current self-reported conditions and a condition within 5 years follow from the hidden imbalance (assumed logistic effects, base risk about 6 to 13%).
- **Face map:** each zone is tied to the body systems in the traditional map (`insights.KB`). The assumed strength of the link is the knob `link`: 0 = the face map has no effect.
- **Scanner noise:** true spots are thinned and polluted with false ones using the *measured* per-zone recall and precision of the real scan (cross-validated on expert-labelled photos),
  so the synthetic counts are as noisy as real scans. Spot counts are calibrated to the real scan (about 10 per photo).
- **Models:** gradient boosting, compared with and without the face-zone variables (30,000 train / 20,000 test people per scenario; bootstrap CIs in `results/simulation_results.json`).

## 1. Does adding face-zone data improve prediction? (test AUC: intake only -> intake + zones)
| Scenario | digestive | endocrine | stress and sleep | mean gain, 8 systems |
|---|---|---|---|---|
| null (link 0.0) | 0.704 -> 0.704 | 0.695 -> 0.695 | 0.711 -> 0.709 | -0.000 |
| weak (link 0.25) | 0.704 -> 0.707 | 0.695 -> 0.694 | 0.711 -> 0.711 | +0.002 |
| moderate (link 0.5) | 0.704 -> 0.710 | 0.695 -> 0.695 | 0.711 -> 0.721 | +0.006 |
| strong (link 0.8) | 0.704 -> 0.722 | 0.695 -> 0.695 | 0.711 -> 0.730 | +0.013 |

Reading: with no real link the zone data add nothing, so the method does not invent a signal. Even if the link were real and strong, the gain is small, because the scanner is noisy and the hidden
imbalance is only partly visible. Most predictive power comes from the health form (history, BMI, stress, sleep), not from the face.

## 2. How many people would a real study need? (power to detect the link, likelihood-ratio test, p < 0.05; 40 simulated studies per cell, so values are approximate)
| System and assumed link | 300 people | 1,000 | 3,000 | 10,000 |
|---|---|---|---|---|
| digestive at link 0.25 | 10% | 20% | 32% | 88% |
| digestive at link 0.5 | 25% | 18% | 78% | 100% |
| stress_sleep at link 0.25 | 12% | 15% | 32% | 80% |
| stress_sleep at link 0.5 | 12% | 30% | 88% | 100% |
| endocrine at link 0.25 | 8% | 2% | 2% | 20% |
| endocrine at link 0.5 | 15% | 2% | 20% | 62% |

Reading: detecting a moderate link needs about 3,000 people for digestive or stress outcomes. The chin and hormonal link is harder to detect because the scan finds chin and nose spots poorly (see zone recall).

## What the app does with this
The trained demonstrator (`models/risk_sim.joblib`, assumed `moderate` link) turns a patient's health form and scan into a *qualitative tier* (higher attention, typical, lower attention) per body system,
relative to the simulated population. The app labels it as a simulated research prototype and never shows a probability.

## Limits
Assumed effect sizes are not measured. The synthetic AUCs describe the method only. Replace the generator with a real cohort (photos or scans, intake form, outcomes at follow-up) to test the claim.

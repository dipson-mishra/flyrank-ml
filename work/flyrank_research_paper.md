# Predicting Search Visibility Decline: A Proactive Triage System for Content Refresh

**Author:** Dipson Mishra
**Date:** 2026-09-12
**Reproducibility:** [GitHub Repo](https://github.com/dipson-mishra/flyrank-ml) | [Capstone Notebook](work/notebooks/capstone.ipynb)

---

## 1. Abstract
Which high-demand search pages are most likely to decline in visibility over the next 30 days? Using the FlyRank Warehouse release (~79M rows), we implemented a Random Forest model trained on a strict temporal out-of-time split. The model achieved a Precision@50 of 0.680, compared to 0.240 for a rule-based baseline. This represents a ~2.8x lift in identifying high-risk candidates at the top of the queue. The resulting system serves as a decision-support tool to prioritize editorial review before traffic loss becomes critical.

## 2. Introduction
Imagine an editor discovering that a primary traffic driver has lost 40% of its visibility overnight. By the time the loss is detected, the opportunity to proactively defend the rank has passed. The business challenge is not just finding "bad" content, but identifying "good" content that is starting to fade.

This research implements a predictive triage system. Instead of reacting to traffic loss, we identify pages with significant current demand that exhibit early behavioral signatures of decline, allowing editorial capacity to be deployed where it can prevent the most significant losses.

## 3. Data
We utilized the FlyRank Warehouse release, accessing approximately 79 million rows of daily performance data. 

**Data Contract:**
- **Feature Window:** 90-day trailing window ending at $T_0$.
- **Label Window:** Subsequent 30-day window ($T_0+1$ to $T_0+30$).
- **Public Safety:** All client-identifying information, URLs, and raw keyword tokens were excluded. Findings are reported as directional decision-support.

## 4. Methodology
To move beyond simple observation, we implemented a **Temporal Predictive Pipeline**. This design ensures the model learns to predict future states rather than memorizing historical correlations.

**The Validation Design:**
We used a strict "Out-of-Time" (OOT) split to simulate real-world deployment:
- **Training Set:** Snapshot from April 30 ($T_0$), predicting May outcomes.
- **Test Set:** Snapshot from May 31 ($T_0$), predicting June outcomes.

**Model Specifications:**
- **Algorithm:** Random Forest Classifier (chosen for non-linear handling of heavy-tailed search distributions).
- **Target:** `is_declining_label = 1` if future impressions dropped significantly below the normalized past average.
- **Features:** Visibility signals (impressions, avg position), engagement (CTR, scroll rate), and content properties (age, word count).
- **Leakage Check:** The strict gap between the feature window and label window prevents temporal leakage.

## 5. Results
The model was evaluated against a transparent rule-based baseline (prioritizing high-visibility pages with current downward trends).

| Metric | Baseline | RF Model | Lift |
| :--- | :--- | :--- | :--- |
| **Precision@50** | 0.240 | 0.680 | $\approx$2.8x |
| **ROC AUC** | - | 0.747 | - |

**Base Rate:** The natural decline rate in the test set was observed, confirming that the model's 0.680 precision is a significant improvement over random chance or simple rules.

**Key Finding:** The strongest predictors were impression volume and average position. This indicates the model successfully identifies "high-value" pages—those that are still visible but are starting to slip.

## 6. Limitations & Honest Framing
This system is a triage tool, not a deterministic crystal ball.

- **Correlation $\neq$ Causation:** We identified signals that precede decline; we cannot prove that an editorial refresh *guarantees* recovery.
- **External Shocks:** Declines caused by search engine algorithm updates or the introduction of AI Overviews may not be predictable via content-level signals.
- **Temporal Drift:** Search behavior evolves. The model requires periodic retraining on the latest temporal windows to remain accurate.

## 7. Ranked Recommendations
The output is a ranked queue of "Refresh Opportunities." We recommend the following triage workflow:

1. **Urgent Review (Prob $\ge$ 0.7):** Inspect immediately. These pages have the highest risk of imminent, significant traffic loss.
2. **Monitor & Review (0.5 $\le$ Prob < 0.7):** Add to the monthly editorial backlog.
3. **Maintain (Prob < 0.5):** No immediate action required; continue standard monitoring.

## 8. Reproducibility
The full pipeline—from DuckDB warehouse extraction to OOT validation—is implemented in `work/notebooks/capstone.ipynb`.

**To reproduce:**
1. Set `HF_TOKEN` environment variable.
2. Execute all cells in `capstone.ipynb`.
3. All constants (split dates, label thresholds) are explicitly defined in the notebook configuration.

## 9. Acknowledgments & Data Credit
This work was made possible using the FlyRank ML Internship dataset provided by [FlyRank](https://flyrank.ai).

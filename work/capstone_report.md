# FlyRank Refresh Opportunity Model (Professional Edition)

- **Author:** Dipson Mishra
- **Lane:** Content refresh prioritization
- **Repo:** `https://github.com/dipson-mishra/flyrank-ml`
- **Date:** 2026-09-12

## 0. Abstract

FlyRank editors face a recurring challenge: identifying pages with significant current demand that are predicted to decline in the near future. We modeled this as a temporal out-of-time predictive task using the FlyRank Warehouse release (~79M rows). By aggregating trailing 90-day behavior to predict decline over the subsequent 30 days, we compared a transparent rule-based baseline against a Random Forest model. The Random Forest model demonstrated superior predictive power on the June test set, achieving a Precision@50 of 0.680 versus 0.240 for the baseline—a roughly 2.8x lift at the top of the queue. The strongest predictors were visibility and engagement signals such as impression volume, average position, and content age, suggesting the model identifies pages with real editorial value that are beginning to slip. This output is intended as a decision-support tool to prioritize editorial review before traffic loss becomes critical.

## 1. Problem framing

The business decision is to prioritize human editorial capacity: among FlyRank pages with visible demand, which ones are most likely to decline in the next 30 days? The unit of analysis is the page, and the output is a ranked refresh-opportunity queue. A wrong call either wastes reviewer time on stable pages or misses a high-volume page that is starting to fade. Machine learning provides a scalable, objective way to identify these high-risk candidates by learning patterns that precede a decline across the entire warehouse dataset.

## 2. Data safety

The project utilizes the FlyRank Warehouse release, accessing data via DuckDB and Hugging Face. We adhered to a strict temporal data contract: features are aggregated over a 90-day window ending at $T_0$, and labels are derived from the subsequent 30-day window ($T_0+1$ to $T_0+30$). This design prevents temporal leakage. We excluded all client-identifying information, page titles, URLs, domains, and raw keyword tokens. All findings are reported as observed, measured, and directional decision-support, ensuring the work remains in the public-safe lane.

## 3. Baseline

The baseline was a transparent, rule-based score that prioritized pages with high visibility but current downward trends (a proxy for risk). While simple and explainable, this baseline relies on current-window observations. When evaluated on the "Out-of-Time" June test set, the baseline produced a Precision@50 of 0.240, serving as a fair benchmark for the predictive model.

## 4. Model / analysis

We implemented a Random Forest Classifier, chosen for its ability to handle the non-linear relationships and heavy-tailed distributions typical of search data. The target was a binary label: `is_declining_label = 1` if future impressions dropped significantly below the normalized past average. The feature set included visibility (impressions, avg position), engagement (CTR, scroll rate, engagement rate), and content properties (age, word count). 

## 5. Evaluation

To ensure true generalization, we used a strict temporal split. The model was trained on a snapshot from April ($T_0 = 2026-04-30$) to predict May outcomes and tested on a snapshot from May ($T_0 = 2026-05-31$) to predict June outcomes. On this "Out-of-Time" test set, the Random Forest achieved a ROC AUC of 0.747 and a Precision@50 of 0.680, significantly outperforming the baseline. This proves the model can identify future decline risk rather than just memorizing past trends.

## 6. Interpretation

The model's strongest drivers were interpretable operational signals: impression volume, average position, and content age. This indicates the model is not reacting to random noise, but is identifying pages that still maintain significant visibility but are exhibiting the early behavioral signatures of a decline. For an editor, this surfaces the "high-value" refresh candidates—pages that are still important enough to save.

## 7. Recommendation

The ranked queue should be used as a triage system for editorial prioritization. The recommended workflow is: inspect the top-ranked candidates first, verify the "predicted decline" against real-world content quality, and decide whether a refresh or re-optimization is warranted. This transforms editorial review from a reactive process into a proactive strategy, reducing the risk of costly traffic loss.

## 8. Reproducibility

The professional implementation is contained entirely within the assignment notebooks in `work/notebooks/`. To reproduce the results:
1. Set the `HF_TOKEN` environment variable.
2. Open `work/notebooks/capstone.ipynb`.
3. Run all cells.

The notebook implements the full pipeline: DuckDB warehouse extraction, temporal windowing, Random Forest training, and out-of-time validation. All constants, including the $T_0$ split dates and the label threshold, are explicitly defined in the code.

## 9. Acknowledgments & data credit

Built on the FlyRank ML Internship dataset, available at: https://flyrank.ai

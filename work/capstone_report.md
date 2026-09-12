# FlyRank Refresh Opportunity Model

- **Author:** Dipson Mishra
- **Lane:** Content refresh prioritization
- **Repo:** `https://github.com/dipson-mishra/flyrank-ml`
- **Date:** 2026-09-12

## 0. Abstract

FlyRank editors commonly face a backlog of pages that still carry visible demand but are quietly slipping in performance, and the team needs a way to identify which ones deserve a refresh review before the traffic drop becomes costly. We modeled this as a client-holdout classification task on an anonymized dataset of 30,000 pages and compared a transparent baseline rule against logistic regression, a decision tree, and a random forest. The random forest gave the strongest Precision@50, reaching 0.680 versus 0.240 for the baseline, a roughly 2.8x lift at the top of the queue. The strongest predictors were engagement and visibility signals such as days with impressions, impression volume, average position, and content age, which suggests the model is finding pages with real editorial value that are beginning to underperform. This output is intended as a review-support tool to prioritize refresh work, not to replace human judgment.

## 1. Problem framing

The business decision is simple: which pages deserve a refresh review before they lose more traffic? The unit of analysis is the page, and the output is a score or queue ranking that prioritizes likely decline risk. A wrong call can either waste reviewer time on low-value pages or miss a high-volume page that has started to fade. Data and machine learning help here because the team needs an objective, repeatable way to surface likely declines in a large content library.

## 2. Data safety

The project uses the bundled anonymized FlyRank dataset, with 30,000 pages and 44 columns. We kept the target definition aligned with the project rules: `is_declining_label` is based on a page’s measured trend direction, while trend and percentage signals are not used as leak-prone features. We excluded client-identifying values and text fields such as page titles, URLs, domains, and keyword tokens from the model inputs. The work stays in the public-safe lane: observed, measured, and decision-support only.

## 3. Baseline

The baseline was a transparent hand-written rules score that sorted likely declining pages based on simple conditions such as decreasing engagement and visible demand. It is useful because it provides a fair comparison against a model and makes the queue easier to explain to editors. On the same evaluation split, the baseline produced a Precision@50 of 0.240.

## 4. Model / analysis

We trained three models: logistic regression, decision tree, and random forest, each using client-holdout validation to avoid row- and client-level leakage across train and test splits. The final model selected on Precision@50 was the random forest. The feature set included page performance, visibility, and engagement metrics such as impression counts, CTR, average position, scroll rate, content age, and days with activity. The target was whether a page is categorized as declining based on tracking behavior over the defined trend window.

## 5. Evaluation

The evaluation used a client-holdout split so that no client’s pages were shared between train and test. On that split, the random forest achieved a ROC AUC of 0.747, average precision of 0.610, Precision@50 of 0.680, recall of 0.741, and F1 of 0.638. The baseline rules trail significantly at 0.240 Precision@50. This is the most relevant comparison for editorial prioritization because the queue is small and the reviewer is interested in the top-ranked slice.

## 6. Interpretation

The highest-weight features were not arbitrary signals; they speak directly to editorial reality. The strongest drivers were days with impressions, impressions over the recent window, average page position, content age, and word count. In plain language, the model identifies pages that still receive real visibility but are trending downward. That is a much more actionable class than a generic low-quality or low-traffic page.

## 7. Recommendation

The ranked queue is best used as a triage system for refresh review. The recommended workflow is: inspect the top of the queue first, check the page in context, and decide whether a refresh, update, or re-optimization is warranted. The model should not replace editorial judgment, but it does provide a much better first pass than a raw backlog or a static threshold.

## 8. Reproducibility

From a fresh clone:

```bash
python -m pip install -r requirements.txt
python -X utf8 scripts/run_all.py
```

This produces the feature vector, baseline queue, ranked queue, evaluation outputs, charts, and the final PDF report under `outputs/`. The results are reproducible with the repo’s committed pipeline and the fixed evaluation setup used in the project.

## 9. Acknowledgments & data credit

Built on the FlyRank ML Internship dataset, available at: https://flyrank.ai

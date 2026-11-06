# Model Card: Application Scorecard v1

## Model details

| | |
|---|---|
| **Name** | Application Scorecard v1 (champion) |
| **Date** | 2026-11-06 |
| **Developer** | Mai Doan Thanh Thuy |
| **Type** | Logistic regression on Weight-of-Evidence (WoE) features, scaled to points |
| **Output** | Credit score (higher = safer), probability of default (PD), decision (Reject / Review / Approve), reason codes |
| **Score scale** | 600 points at good:bad odds 50:1; every 20 points (PDO) doubles the odds |
| **Artifacts** | `outputs/model_features.csv`, `woe_spec_final.pkl`, `scorecard.csv`, `score_scaling.csv` |
| **Code** | `src/pipeline.py` (scoring), `src/cli.py` (command line) |
| **Challengers** | LightGBM on the same 16 features (with and without monotone constraints) and on all 119 features. These are benchmarks only and are not used for decisions. |

## Intended use

- **Primary use:** decide on consumer loan applications at origination by ranking applicants by default risk. Applicants are rejected automatically, routed to manual review, or approved automatically.
- **Users:** credit risk analysts and underwriters. The reason codes support underwriters in the Review band and the adverse-action notices sent to rejected applicants.
- **Out of scope:**
  - Behavioural scoring of existing customers.
  - Pricing or limit setting without separate validation.
  - Populations or products other than the Home Credit cash and revolving loans in the training data.
  - Fully automated decisions on the Review band.

## Training data

- **Source:** Home Credit Default Risk (Kaggle), `application_train.csv`, 307,511 applications × 122 columns. Only the application table is used, with no bureau or previous-application data.
- **Target:** `TARGET = 1` if the client had payment difficulties. Default rate 8.07% (11.4 : 1).
- **Split:** stratified random 80/20. Train has 246,008 applications and test has 61,503, both with a default rate of 8.07%. All binning, WoE values and model fitting use train only.
- **Cleaning:**
  - `DAYS_EMPLOYED = 365243` (18% of rows, mostly pensioners) is set to missing and kept as a flag.
  - `XNA` and `Unknown` labels are set to missing.
  - Day offsets are converted to years.
  - Missing values are **not imputed**: they form their own bin.

## Features

16 features selected by information value (IV ≥ 0.02), correlation (|r| < 0.7), coefficient sign, p-value and VIF (all VIFs < 2). The points range is the weight of the feature in the score, over bins holding at least 0.5% of applicants.

| Feature | Description | Points range |
|---|---|---|
| `EXT_SOURCE_3` | External credit score 3 | 49 |
| `EXT_SOURCE_2` | External credit score 2 | 42 |
| `EXT_SOURCE_1` | External credit score 1 | 31 |
| `NAME_EDUCATION_TYPE` | Education level | 12 |
| `YEARS_EMPLOYED` | Years in current employment | 9 |
| `OWN_CAR_AGE` | Age of own car (missing = no car) | 9 |
| `OCCUPATION_TYPE` | Occupation | 9 |
| `ORGANIZATION_TYPE` | Employer type | 8 |
| `REGION_RATING_CLIENT_W_CITY` | Region rating, including city | 7 |
| `FLAG_DOCUMENT_3` | Provided document 3 | 7 |
| `AMT_GOODS_PRICE` | Price of the goods financed | 6 |
| `FLOORSMAX_AVG` | Max floors of the client's building (normalised) | 6 |
| `YEARS_REGISTRATION` | Years since registration change | 4 |
| `REG_CITY_NOT_LIVE_CITY` | Registered city differs from living city | 4 |
| `YEARS_LAST_PHONE_CHANGE` | Years since last phone change | 4 |
| `YEARS_ID_PUBLISH` | Years since ID document change | 2 |

The three external scores account for most of the score. Within every feature, points fall as the bad rate rises.

**Excluded on purpose:**
- `CODE_GENDER` and `NAME_FAMILY_STATUS` are protected attributes.
- `AGE_YEARS` was dropped because its coefficient flipped sign, owing to overlap with `EXT_SOURCE_1` and `YEARS_EMPLOYED`.

## Performance (test set)

| Metric | Value | Reference |
|---|---|---|
| AUC | 0.739 | |
| Gini | 0.477 (95% CI 0.464–0.490) | > 0.4 acceptable for retail scorecards |
| KS | 0.357 | > 0.3 acceptable |
| PR-AUC | 0.216 | random = 0.081 (2.7×) |
| Train − test Gini gap | −0.002 | no over-fitting |
| Calibration (observed / predicted, by decile) | 0.91–1.08 | 1 = perfect |
| Top decile lift | 3.1 | |
| Bads captured in the riskiest 30% | 62% | |
| Score PSI / max feature CSI (train vs test) | 0.0002 / 0.0004 | < 0.1 stable |

**Versus challengers.** LGBM-full reaches Gini 0.518. Its gain over the scorecard is +0.041, with a paired-bootstrap 95% CI of 0.035–0.048. The gain is statistically significant but small, and it comes with 119 features, the use of age, and local explanations that are harder to defend. The scorecard is retained as champion.

## Decision policy

| Decision | Score | PD | Share of applicants | Default rate | Share of all bads |
|---|---|---|---|---|---|
| Reject | < 540 | > 13.8% | 15.8% | 21.5% | 42% |
| Review | 540–559 | 7.4–13.8% | 23.4% | 10.1% | 29% |
| Approve | ≥ 560 | ≤ 7.4% | 60.7% | 3.8% | 28% |

Applicants are approved automatically only when their PD is below the portfolio average (8.07%). Compared with a single cut-off at 550, the share of all bads that are approved automatically falls from 42% to 28%. If manual-review capacity is limited, the 545 / 555 scheme sends 12% of applicants to review, and 35% of bads are approved automatically. Thresholds are set in `src/pipeline.py`.

## Explainability

- The score is the sum of the points of the applicant's bins. Each feature's **contribution** is its points minus the training average. These contributions are the exact SHAP values of the scorecard and satisfy `average score (567) + sum of contributions = score`.
- **Reason codes** are the largest negative contributions; `score_applicant` returns the top 4. The CLI prints them and saves a waterfall chart.
- **Caveat:** `FLAG_DOCUMENT_3 = 1` lowers the score. The effect is data-driven but hard to explain to a customer. Customer-facing reason texts should be reviewed before use.

## Fairness

Protected attributes are excluded, but outcomes differ by group (test set):

| Group | Default rate | Calibration (observed / predicted) | Score bias (approx.) | Reject / Review / Approve |
|---|---|---|---|---|
| Women | 7.0% | 0.93 | ~2 points too low | 14% / 22% / 64% |
| Men | 10.2% | 1.12 | ~4 points too high | 20% / 26% / 54% |
| Under 25 | 11.7% | 0.74 | ~10 points too low | 49% / 33% / 18% |
| 55+ | 5.4% | | | 5% / 16% / 80% |

- **Gender:** the features predict gender with AUC 0.88 (proxies include `OCCUPATION_TYPE`, `OWN_CAR_AGE` and `ORGANIZATION_TYPE`). The resulting bias is small in points. Women still score higher than men on average, because their risk is genuinely lower. The approval ratio of men to women is 0.89, which passes the 4/5 rule.
- **Age:** this is the material issue. Without age, proxies over-penalise young applicants. Their true risk corresponds to a Review-band score, yet half of them are rejected automatically, where no underwriter sees them. **Recommended policy rule (not implemented):** route under-25 applicants scoring 530–559 to Review. An alternative is to reintroduce age where regulation permits; LGBM-full, which uses age, has calibration 0.93 for this group.
- Ranking power (AUC) is similar across genders and lowest for applicants aged 55+ (0.69–0.71).

Score biases are approximations: each group's mean predicted PD and its observed default rate are converted to the score scale.

## Limitations and risks

- **No out-of-time validation.** Train and test come from a random split of the same period. The near-zero PSI/CSI says nothing about stability over time.
- **No reject inference.** The model learned only from approved loans. Applicants similar to those rejected in the past may be mis-scored.
- **Limited data:** application table only. Models using bureau data reach AUC ≈ 0.80.
- **Dependence on external scores.** If the `EXT_SOURCE_*` providers change their methodology or coverage, the score shifts. These features need their own monitoring.
- **Small subgroups.** The under-25 group has 2,390 test applicants and 280 defaults, so group metrics have wide uncertainty.

## Deployment considerations

The repository is a validated prototype. A production deployment would add:

1. **Service layer.** A scoring API wrapping `score_applicant`, with raw inputs passed through `clean_application`. Range checks are needed, for example `EXT_SOURCE_*` in [0, 1] and non-negative durations. Every request should log the inputs, score, decision, reasons and model version.
2. **Policy rules before the score:** eligibility (age, residency), fraud and blacklist checks, and the under-25 Review rule.
3. **Versioned artifacts.** The four files listed under *Artifacts* form one model version and must be deployed together.
4. **Champion–challenger.** Run LGBM-16-mono in shadow mode to collect evidence for a v2.
5. **Governance.** Independent validation and sign-off, using notebooks 07–13 as documentation, before go-live and after any change.

**Monitoring plan**

| What | Frequency | Trigger |
|---|---|---|
| Score PSI, feature CSI | Monthly | > 0.1 investigate, > 0.25 consider redevelopment |
| Band volumes (Reject / Review / Approve) | Monthly | Large drift from 16 / 23 / 61% |
| Underwriter overrides in Review | Quarterly | Systematic overrides in one direction |
| Default rate by band, Gini, calibration | From 12 months on books | Gini drop or decile calibration outside 0.8–1.2 |
| Approval and calibration by age and gender | Quarterly | Calibration < 0.8 or disparate impact < 0.8 |

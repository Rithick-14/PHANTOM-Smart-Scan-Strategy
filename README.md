# Smart Scan Strategy for Electronic Warfare

A machine-learning-based Electronic Support (ES) receiver scheduler software simulation developed for SIH.

## Overview
This project simulates an RF environment and demonstrates that an integrated adaptive scheduler (**Contextual Bandits + Dynamic RF Knowledge Map + Uncertainty-Aware Exploration + Environment Change Detection**) can optimize scan decisions over traditional fixed/sequential scan strategies.

## Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Launch Dashboard
```bash
streamlit run app/streamlit_app.py
```

## Architecture Phases
- [x] **Phase 0**: Requirements & Architecture Specification
- [x] **Phase 1**: Minimal RF Environment Simulator (`smart_ew.environment`)
- [x] **Phase 2**: Receiver Model (`smart_ew.receiver`)
- [x] **Phase 3**: Sequential Baseline Scheduler (`smart_ew.scheduling.sequential`)
- [x] **Phase 4**: Evaluation Engine (`smart_ew.evaluation`)
- [x] **Phase 5**: Feature Engineering (`smart_ew.features`)
- [x] **Phase 6**: ML Activity Predictor (`smart_ew.models`)
- [x] **Phase 7**: Dynamic RF Knowledge Map (`smart_ew.knowledge`)
- [x] **Phase 8**: Contextual Bandit Scheduler (`smart_ew.scheduling.contextual_bandit`)
- [x] **Phase 9**: Uncertainty-Aware Exploration
- [x] **Phase 10**: Environment Change Detection
- [x] **Phase 11**: Periodic Emitter Model
- [x] **Phase 12**: Frequency-Agile Emitter Model
- [x] **Phase 13**: Multi-Objective Reward Engine
- [x] **Phase 14**: Streamlit Dashboard
- [x] **Phase 15**: Ablation Experiments
- [x] **Phase 16**: Final Benchmarks
- [x] **Phase 17**: Documentation & Presentation Materials
- [x] **Phase 18**: Advanced Extensions (Deep RL Scheduler & FastAPI REST Microservice)

---

## 10-Step Incremental Development Roadmap & Status

### [x] STEP 1 — RF Environment Simulator
- **Objective:** Build synthetic RF spectrum world maintaining unobservable ground truth activity.
- **Files:** [`schema.py`](file:///c:/Users/TKM00630/OneDrive/Desktop/SIH/src/smart_ew/environment/schema.py), [`simulator.py`](file:///c:/Users/TKM00630/OneDrive/Desktop/SIH/src/smart_ew/environment/simulator.py), [`continuous.py`](file:///c:/Users/TKM00630/OneDrive/Desktop/SIH/src/smart_ew/emitters/continuous.py), [`intermittent.py`](file:///c:/Users/TKM00630/OneDrive/Desktop/SIH/src/smart_ew/emitters/intermittent.py)
- **Verification:** `python -m pytest tests/test_environment.py` (5/5 Passed)
- **Demo Command:** `python experiments/run_step1_demo.py`
- **Output:** Generates 2D time-frequency activity matrix $(10 \times 100)$, power matrix ($\text{dBm}$), ASCII map, and graphic heatmap [`experiments/results/step1_spectrum_activity.png`](file:///c:/Users/TKM00630/OneDrive/Desktop/SIH/experiments/results/step1_spectrum_activity.png).

---

### [x] STEP 2 — Receiver Simulator
- **Objective:** Build software model of ES receiver with instantaneous bandwidth limits, retuning penalties, thermal noise floor, SNR thresholds, and probabilistic ROC curve detection.
- **Files:** [`schema.py`](file:///c:/Users/TKM00630/OneDrive/Desktop/SIH/src/smart_ew/receiver/schema.py), [`model.py`](file:///c:/Users/TKM00630/OneDrive/Desktop/SIH/src/smart_ew/receiver/model.py), [`detection/model.py`](file:///c:/Users/TKM00630/OneDrive/Desktop/SIH/src/smart_ew/detection/model.py)
- **Verification:** `python -m pytest tests/test_receiver.py` (4/4 Passed)
- **Demo Command:** `python experiments/run_step2_demo.py`
- **Output:** Encapsulated `ReceiverObservation` generator enforcing zero data leakage, retuning time overhead ($\tau_{\text{retune}} = 1.5$), and $P_{\text{fa}} = 2.0\%$.

---

### [x] STEP 3 — Fixed / Sequential Scan Baselines
- **Objective:** Build conventional open-loop scan strategies (Sequential Sweep $0 \to N-1 \to 0$ and Uniform Random Sweep) as benchmark baselines.
- **Files:** [`base.py`](file:///c:/Users/TKM00630/OneDrive/Desktop/SIH/src/smart_ew/scheduling/base.py), [`sequential.py`](file:///c:/Users/TKM00630/OneDrive/Desktop/SIH/src/smart_ew/scheduling/sequential.py), [`random_scan.py`](file:///c:/Users/TKM00630/OneDrive/Desktop/SIH/src/smart_ew/scheduling/random_scan.py)
- **Verification:** `python -m pytest tests/test_schedulers.py` (4/4 Passed)
- **Demo Command:** `python experiments/run_step3_demo.py`
- **Output:** Quantitative baseline demonstration showing low intercept rates ($\sim 9.14\%$ for Sequential, $\sim 13.14\%$ for Random) due to blind scanning.

---

### [x] STEP 4 — Performance Measurement & Metrics Harness
- **Objective:** Implement standardized 10-metric evaluation suite ($P_d$, $P_{\text{fa}}$, Intercept Rate, Average Intercept Time, Intercept Delay Std, Spectrum Coverage, Retuning Cost, Cumulative Reward).
- **Files:** [`metrics.py`](file:///c:/Users/TKM00630/OneDrive/Desktop/SIH/src/smart_ew/evaluation/metrics.py), [`runner.py`](file:///c:/Users/TKM00630/OneDrive/Desktop/SIH/src/smart_ew/evaluation/runner.py)
- **Verification:** `python -m pytest tests/test_evaluation.py` (2/2 Passed)
- **Demo Command:** `python experiments/run_step4_demo.py`
- **Output:** Automated closed-loop simulation benchmark comparing all strategies across full 10-metric table.

---

### [x] STEP 5 — ML Activity Prediction
- **Objective:** Extract 10D contextual features and train supervised machine learning models to predict $P(\text{Emitter Activity} \mid \text{History})$.
- **Files:** [`schema.py`](file:///c:/Users/TKM00630/OneDrive/Desktop/SIH/src/smart_ew/features/schema.py), [`extractor.py`](file:///c:/Users/TKM00630/OneDrive/Desktop/SIH/src/smart_ew/features/extractor.py), [`predictor.py`](file:///c:/Users/TKM00630/OneDrive/Desktop/SIH/src/smart_ew/models/predictor.py)
- **Verification:** `python -m pytest tests/test_features.py tests/test_models.py` (6/6 Passed)
- **Demo Command:** `python experiments/run_step5_demo.py`
- **Output:** Model comparison (Logistic Regression $80\%$, Random Forest $90\%$, Gradient Boosting $90\%$) and online calibrated probability predictions matching ground truth active bands ($92\%$ on Band 2, $80\%$ on Band 5).

### [x] STEP 6 — Smart Scheduler & Dynamic Knowledge Map
- **Objective:** Implement central state representation (`DynamicRFKnowledgeMap`) and score-based multi-objective scheduler balancing activity probability, recency deltas, Bayesian belief variance, and retuning cost.
- **Files:** [`schema.py`](file:///c:/Users/TKM00630/OneDrive/Desktop/SIH/src/smart_ew/knowledge/schema.py), [`map.py`](file:///c:/Users/TKM00630/OneDrive/Desktop/SIH/src/smart_ew/knowledge/map.py), [`uncertainty_aware.py`](file:///c:/Users/TKM00630/OneDrive/Desktop/SIH/src/smart_ew/scheduling/uncertainty_aware.py)
- **Verification:** `python -m pytest tests/test_knowledge_map.py tests/test_uncertainty_aware.py` (6/6 Passed)
- **Demo Command:** `python experiments/run_step6_demo.py`
- **Output:** Dynamic Bayesian knowledge state updates ($\mu_b, \sigma_b, C_b$) and performance comparison showing proposed Smart Scheduler achieving $28.86\%$ Intercept Rate vs $8.96\%$ for Sequential Sweep.

### [x] STEP 7 — Uncertainty & Exploration Engine
- **Objective:** Implement multi-source uncertainty estimation (Bayesian variance, ensemble disagreement, observation count sparsity) and exploration policies (UCB, Epsilon-Greedy, Softmax).
- **Files:** [`exploration.py`](file:///c:/Users/TKM00630/OneDrive/Desktop/SIH/src/smart_ew/scheduling/exploration.py), [`test_exploration.py`](file:///c:/Users/TKM00630/OneDrive/Desktop/SIH/tests/test_exploration.py)
- **Verification:** `python -m pytest tests/test_exploration.py` (3/3 Passed)
- **Demo Command:** `python experiments/run_step7_demo.py`
- **Output:** Quantitative proof showing UCB Explorer discovering late-appearing emitters within $+2$ time steps vs complete starvation under Pure Greedy scanning and $+51$ steps delay under Epsilon-Greedy.

### [x] STEP 8 — Periodic & Frequency-Agile Emitters
- **Objective:** Extend simulator and predictor to model periodic pulsed radar timing ($T, d, \phi$) and frequency-agile Markov transition matrices $P(\text{Band}_j \mid \text{Band}_i)$.
- **Files:** [`periodic.py`](file:///c:/Users/TKM00630/OneDrive/Desktop/SIH/src/smart_ew/emitters/periodic.py), [`agile.py`](file:///c:/Users/TKM00630/OneDrive/Desktop/SIH/src/smart_ew/emitters/agile.py), [`periodicity.py`](file:///c:/Users/TKM00630/OneDrive/Desktop/SIH/src/smart_ew/knowledge/periodicity.py), [`transition_predictor.py`](file:///c:/Users/TKM00630/OneDrive/Desktop/SIH/src/smart_ew/knowledge/transition_predictor.py)
- **Verification:** `python -m pytest tests/test_periodic.py tests/test_agile.py` (6/6 Passed)
- **Demo Command:** `python experiments/run_step8_demo.py`
- **Output:** Periodic pulse period estimation ($T=10$, $P(t=60)=95\%$) and frequency-hopping transition matrix learning ($P(\text{Band 5} \mid \text{Band 1}) = 100\%$).

---

### [x] STEP 9 — Interactive Dashboard
- **Objective:** Build Streamlit web application visualizing real-time spectrum heatmaps, receiver scan trajectories, probability/uncertainty maps, KPI cards, and comparative benchmark metrics for judges.
- **Files:** [`streamlit_app.py`](file:///c:/Users/TKM00630/OneDrive/Desktop/SIH/app/streamlit_app.py), [`anomaly_detector.py`](file:///c:/Users/TKM00630/OneDrive/Desktop/SIH/src/smart_ew/knowledge/anomaly_detector.py), [`run_step9_demo.py`](file:///c:/Users/TKM00630/OneDrive/Desktop/SIH/experiments/run_step9_demo.py)
- **Launch Command:** `streamlit run app/streamlit_app.py`
- **Output:** Interactive dark-themed web UI featuring Plotly waterfall heatmaps, debug ground-truth toggle, 5 KPI cards (calibrated non-100% $P_d$ & temporal duty-cycle coverage), real-time comparative line charts, and a dedicated **Real-Time RF Anomaly Detection Dashboard Table** flagging environment drift, unexpected bursts, power spikes, and agile pattern shifts.

---

### [x] STEP 10 — Final Benchmarking & Ablation Study
- **Objective:** Conduct multi-seed evaluation across 10 scenarios and execute systematic component ablation study (Model A through F) quantifying individual algorithmic contributions.
- **Files:** [`run_benchmarks.py`](file:///c:/Users/TKM00630/OneDrive/Desktop/SIH/experiments/run_benchmarks.py), [`run_ablation.py`](file:///c:/Users/TKM00630/OneDrive/Desktop/SIH/experiments/run_ablation.py), [`run_step10_demo.py`](file:///c:/Users/TKM00630/OneDrive/Desktop/SIH/experiments/run_step10_demo.py)
- **Verification:** `python -m pytest tests/test_benchmarks.py tests/test_ablation.py` (4/4 Passed)
- **Demo Command:** `python experiments/run_step10_demo.py`
- **Output:** Automated JSON/Markdown benchmark reports saved at [`data/processed/benchmark_report.md`](file:///c:/Users/TKM00630/OneDrive/Desktop/SIH/data/processed/benchmark_report.md) demonstrating proposed integrated framework performance vs baselines.

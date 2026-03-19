# deep-performance-rul

Code accompanying the paper *“A Deep Learning Approach for Performance-Based Prediction of Remaining Useful Life”*.  

This repository implements a performance-based RUL prediction framework that preserves an explicit estimation–prediction structure, combining latent health index estimation with particle filter–based uncertainty-aware degradation modeling.


## Qualitative Results

<div align="center">
  <img src="https://github.com/4belito/deep-performance-rul/blob/main/assets/rul_pred_example.gif?raw=true" width="5200">
  <p><em>Example RUL prediction</em></p>
</div>


## Installation

This project uses [`uv`](https://github.com/astral-sh/uv) for dependency management.

### 1. Install uv

**macOS (recommended if you use Homebrew):**
```bash
brew install uv
```

**macOS or Linux (if you do not use Homebrew):**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows (PowerShell):**
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Verify installation:
```bash
uv --version
```

---

### 2. Clone the repository

```bash
git clone https://github.com/4belito/deep-performance-rul.git
cd deep-performance-rul
```

---

### 3. Install dependencies

```bash
uv sync
```

---
## Running Experiments

The following steps reproduce the results reported in the paper.

---

### 0. Experiment Configuration

All experiment settings are defined in `experiment_config.py`.

The default values correspond to those used in the paper.  
To reproduce the reported results, select the desired dataset in this file and keep all other parameters unchanged.

---

### 1. Data Preparation

Run the notebook:

```
1-data_preparation.ipynb
```

Execute it for both data splits:

- `data_split = "dev"`
- `data_split = "test"`

Make sure the `deep-performance-rul` kernel (uv environment) is selected.

---

### 2. Train Operation Condition Normalization Network

Run:

```
2-train_ocnorm.ipynb
```

This trains the operation condition normalization model.

---

### 3. Apply Operation Condition Normalization (Estimation Step)

Run:


```
3-estimation_data.ipynb
```

Execute it for both:

- `data_split = "dev"`
- `data_split = "test"`

This step:

- Applies operation condition normalization  
- Normalizes performance metrics to the interval `[0, 1]`  
- Removes metrics that do not exhibit a valid degradation pattern  
- Removes training units for which none of the performance metrics approach their EOL

The selected performance metrics are displayed before saving the processed data.

---

### 4. Train Degradation Models (Offline Units)

Run:

```
4-degmodel_train.ipynb
```

This trains one degradation model per development unit. These models are used:

- To initialize the particle filter state  
- As priors in the particle filter correction step  

Optional visualization notebooks:

- `4p-init_states_plots.ipynb` — Plots the learned stochastic degradation processes  
- `4p-mixture_model_plot.ipynb` — Plots the learned mixture model (initial PF prior)

---

### 5. Train Particle Filter Controller Network

Run:

```
5-pf_net_train.ipynb
```

Train the controller network separately for each performance metric selected in Step 3.

Execute the notebook once for each retained performance metric, for example:

- `perform_name = "T48"`


Optional:

- `5v-pf_test_video.ipynb` — Generates a video of EOL prediction on test data  

---

### 6. RUL Prediction

Run:

```
6-rul_test_rep.ipynb
```

This predicts RUL on the test set for `N_REP` repetitions  
(default: `N_REP = 10`, configurable in `experiment_config.py`).

Optional:

- `6p-rul_test_rep_plot.ipynb` — RUL prediction plots  
- `6v-rul_test_video.ipynb` — RUL prediction video  

---

### 7. Evaluation

Run:

```
7-results.ipynb
```

This notebook:

- Averages predictions across the `N_REP` runs  
- Computes evaluation metrics in three regions:

1. **Full lifetime**  
2. **Last 65 time steps** (RUL ≤ 65), for comparison with clipped methods  
3. **Degradation region**, defined by the health-state (`hs`) variable  

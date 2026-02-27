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

### 3. Apply Operation Condition Normalization

Run:

```
3-apply_ocnorm.ipynb
```

Execute it for both:

- `data_split = "dev"`
- `data_split = "test"`

This step:

- Applies operation condition normalization.
- Normalizes performance metrics to the interval $[0,1]$.
- Applies a causal filter to the performance signals.
- Removes performance metrics that do not exhibit a valid degradation pattern after normalization.
- Removes training units for which none of the selected performance metrics approach end-of-life behavior after normalization.

The selected performance metrics are displayed before saving the processed data.

---

### 4. Train Degradation Models (Offline Units)

Run:

```
4-degmodel_train.ipynb
```

This trains one degradation model per development unit. These models are used:

- To initialize the particle filter state  
- As a prior in the particle filter correction step  

Optional visualization notebooks:

- `4p-init_states_plots.ipynb`
- `4p-mixture_model_plot.ipynb`

These provide diagnostic plots of the degradation model fits and the corresponding mixture model.

---

### 5. Train Particle Filter Controller Network

Run:

```
5-pf_net_train.ipynb
```

Train the controller network separately for each performance metric selected in Step 3.

Only performance metrics retained after the normalization and filtering procedure should be used.

---

### 6. Controller Network Evaluation (Optional)

The following notebooks provide qualitative evaluation:

- `6-pf_test_video.ipynb` - Network evalaution on testing data
- `6-pf_eval_video.ipynb` - Network evaluaiton on trainin data
- '6-plot_net_otput.ipynb' - Network output visualization notebook

Evaluation protocol:
- Use $n-1$ development units as offline units  
- Predict on the remaining unit  

These notebooks generate videos and diagnostic plots of EOL prediction behavior.

---

### 7. RUL Prediction

Run:

```
7-rul_avgtest.ipynb
```

This predicts RUL on the test set for `N_REP` repetitions (default: `N_REP = 10`, configurable in `experiment_config.py`).

Additional notebooks:

- `7-rul_eval.ipynb`
- `7-rul_test.ipynb`

These generate videos for evaluation and test RUL predictions.

---

### 8. Compute Evaluation Metrics

Run:

```
8-results_avg.ipynb
```

This notebook:

- Averages predictions across the `N_REP` runs  
- Computes evaluation metrics in three regions:

1. **Full lifetime**
2. **Last 65 time steps** (where true RUL ≤ 65), for comparison with clipped state-of-the-art methods
3. **Degradation region**, defined by the health state (`hs`) variable

The results reported in the paper are included as markdown tables.  
If experiment configurations are unchanged, the computed results should match those reported.

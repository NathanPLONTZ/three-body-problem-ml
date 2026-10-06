# Three-Body Problem with Machine Learning

Predicting the trajectories of three bodies attracting each other, using classical machine learning models instead of a numerical solver.

This project was developed for the Machine Learning course (Lisbon) and for the Kaggle community competition
[**Machine Learning@NOVA 2025: the three-body problem**](https://www.kaggle.com/competitions/machine-learning-nova-2025-the-three-body-problem),
hosted by Claudia Soares.

## The problem

The three-body problem consists of taking the initial positions and velocities of three bodies and solving their motion according to Newton's laws.
Unlike the two-body problem, it has no closed-form solution: the system is generally chaotic (non-periodic) and is usually handled with numerical integration, which is computationally expensive and time-consuming.

**Goal of the challenge:** learn to predict the movement of the three bodies on a 2D plane, given their initial positions (the initial velocities are zero), **without using numerical solvers**.

- **Metric:** Root Mean Squared Error (RMSE) between the predicted position coordinates and the observed targets.
- **Submission:** a CSV with one row per `Id` and the columns `Id, x_1, y_1, x_2, y_2, x_3, y_3`.

## Data

The data comes from simulations of the three-body problem. It is not included in this repository (about 500 MB): download it from the
[competition's Data tab](https://www.kaggle.com/competitions/machine-learning-nova-2025-the-three-body-problem/data) and place the three CSV files in the `data/` folder.

| File | Description |
|------|-------------|
| `data/X_train.csv` | Simulated trajectories: position and velocity of each body over time. |
| `data/X_test.csv` | Initial positions and the time steps at which the positions must be predicted. |
| `data/sample_submission.csv` | Example of a valid submission (random predictions). |

Columns:

- **`X_train.csv`**: `t` (time step), `x_i`, `y_i` (position of body `i`), `v_x_i`, `v_y_i` (velocity of body `i`) for `i` in 1..3, and `Id`.
- **`X_test.csv`**: `Id`, `t`, and the initial positions `x0_i`, `y0_i` of the three bodies (positions at `t = 0`).
- **`sample_submission.csv`**: `Id`, `x_i`, `y_i` for `i` in 1..3.

The training set contains 1,285,000 rows: 5,000 trajectories of 257 consecutive rows each (`t = 0` to `t = 10`, so one row is 10/257 s).
Two trajectories are made only of zeros and some others contain a collision between bodies, after which the values are zero.
Cleaning removes the all-zero trajectories and truncates the others at the first collision (4,998 trajectories and 1,089,790 rows remain).
It also checks that there are no missing values, no infinite values and no empty columns.

## Approach

All the models predict the positions at time `t` from `t` and the **initial positions only** (the velocities are not available at test time).
Whole trajectories are split at random into train (60%), validation (20%) and test (20%) sets, so that no trajectory is shared between sets.
No cross-validation is used for this split: the dataset is large enough, and it would take too much time.

| Notebook | Model |
|----------|-------|
| [task1.ipynb](src/task1.ipynb) | Linear regression baseline (standardized inputs) |
| [task2.ipynb](src/task2.ipynb) | Polynomial regression with L2 (ridge) regularization, degree and regularization strength chosen on the validation set |
| [task3.ipynb](src/task3.ipynb) | Variable removal, then polynomial regression with physics-inspired features and a search over feature groups |
| [task4.ipynb](src/task4.ipynb) | K-nearest neighbors regression, with and without physics-inspired features |

The physics-inspired features (see `add_three_body_features` in [src/utils.py](src/utils.py)) are grouped as follows:

| Group | Features |
|-------|----------|
| 0 (always included) | `t`, `x_1`, `y_1`, `x_2`, `y_2` |
| 1 | pairwise distances `r_12`, `r_13`, `r_23` |
| 2 | inverse distances `inv_r_12`, `inv_r_13`, `inv_r_23` |
| 3 | distance ratios `r12_over_r13`, `r12_over_r23`, `r13_over_r23` |
| 4 | area of the triangle formed by the three bodies |
| 5 | distances to the center of mass `d1_cm`, `d2_cm`, `d3_cm` |

The function also computes the internal angles of the triangle and an approximate angular momentum. All the combinations of groups are tested and compared by RMSE.

## Results

Average RMSE on the local test set, over several random splits:

| Model | Average RMSE |
|-------|--------------|
| Linear regression (baseline) | 1.41 (1.43 on Kaggle) |
| Polynomial regression, degree 2 with regularization | 1.33 |
| Polynomial regression without `x_3`, `y_3` | 1.30 |
| Polynomial regression with selected features (groups 0 + 1 + 3) | 1.18 |
| k-NN with selected features (groups 0 + 3) | **0.89** |

On the Kaggle private leaderboard, the final score is **0.94558** (rank 10). The RMSE values obtained locally are close to the Kaggle ones, so the local validation is consistent.
The values saved in the notebook outputs come from single runs, so they can differ slightly from these averages.

Main findings:

- **Linear baseline:** the linear model gets an RMSE of about 1.41 and leaves a lot of the dynamics unexplained.
- **Polynomial degree:** the number of polynomial features grows quickly with the degree (35 features at degree 2, 791 at degree 5, more than 170,000 at degree 15 for 7 inputs), which makes high degrees slow and prone to overfitting.
  Over 15 runs on small subsets of 50 trajectories, the best degree was 1 without regularization (RMSE 1.20) and 2 with regularization (RMSE 1.14). Ridge and Lasso gave similar results, and Ridge was kept because it is much faster.
- **Removing variables:** `x_3` and `y_3` are strongly correlated with the other coordinates. Removing them does not change the RMSE significantly (1.32 to 1.30), so they are dropped to reduce redundancy.
- **Adding variables:** adding features improves the RMSE over the baseline. Groups 1 and 3 are always among the best combinations. With degree 2, only combinations of up to 3 groups could be tested (too many features otherwise):

  | Groups | RMSE |
  |--------|------|
  | (0, 1, 2) | 1.1857 |
  | (0, 2, 5) | 1.1900 |
  | (0, 1, 3) | 1.1907 |
  | (0, 1, 5) | 1.1911 |
  | (0) only | 1.2934 |

  Groups 0 + 1 + 3 were kept: their RMSE is almost the same as the two best ones, and they improve the polynomial model by about 0.1.
- **k-NN:** the RMSE decreases steadily with `k` and the best value tested is `k = 14`, with no visible overfitting, probably because the dataset is large. Training time is almost constant (k-NN only stores the training points), while inference time grows with `k`.
  The best feature groups for k-NN are different from the polynomial models: 0 + 3, and 0 + 3 + 5.

What was difficult: long training times for high polynomial degrees, some instability of the k-NN results between runs, and understanding the structure of the data (one trajectory is 257 rows).

## Repository structure

```
.
├── data/                     # put the Kaggle CSV files here (not versioned)
├── src/
│   ├── data_exploration.ipynb  # first look at the data
│   ├── data_cleaning.ipynb     # removal of invalid trajectories and sanity checks
│   ├── task1.ipynb             # linear regression baseline
│   ├── task2.ipynb             # polynomial regression
│   ├── task3.ipynb             # polynomial regression + engineered features
│   ├── task4.ipynb             # KNN regression
│   └── utils.py                # shared helpers (cleaning, splitting, features, models, plots)
├── requirements.txt
└── README.md
```

### `src/utils.py`

| Function | Purpose |
|----------|---------|
| `dataset_summary`, `check_dataset_anomalies` | Statistics, boxplots, histograms and sanity checks |
| `clean_data` | Removes all-zero trajectories and truncates trajectories at the first collision |
| `split_trajectories` | Splits whole trajectories into train, validation and test sets |
| `replicate_initial_position_by_block` | Copies the initial positions onto every row of a trajectory |
| `get_n_trajectories` | Returns the first `n` trajectories of a dataset |
| `add_three_body_features` | Adds the physics-inspired features |
| `validate_poly_regression` | Polynomial regression (none / L1 / L2), keeps the best model on validation |
| `validate_knn_regression` | KNN regression for several values of `k`, with RMSE and timing plots |
| `plot_y_yhat` | True vs. predicted positions |

## Getting started

1. Install the dependencies (Python 3.10+ recommended):

   ```bash
   pip install -r requirements.txt
   ```

2. Download the data from Kaggle and put `X_train.csv`, `X_test.csv` and `sample_submission.csv` in `data/`.

3. Run the notebooks from the `src/` folder (they import `utils.py` and read `../data/`). A suggested order:
   `data_exploration` → `data_cleaning` → `task1` → `task2` → `task3` → `task4`.

Each task notebook writes its submission file (for example `baseline-model.csv` or `knn_submission.csv`) in `src/`.

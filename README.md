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

The training set contains 5,000 trajectories of 257 consecutive rows each (`t = 0` to `t = 10`). Some trajectories are invalid and are removed during cleaning:
trajectories made only of zeros, and the zeros that follow a collision between bodies.

## Approach

All the models predict the positions at time `t` from `t` and the **initial positions only** (the velocities are not available at test time).
Whole trajectories are split into train (60%), validation (20%) and test (20%) sets, so that no trajectory is shared between sets.

| Notebook | Content | Reported RMSE (local split) |
|----------|---------|-----------------------------|
| [task1.ipynb](src/task1.ipynb) | Linear regression baseline | 1.42 (test) |
| [task2.ipynb](src/task2.ipynb) | Polynomial regression (degree 2) with L2 (ridge) regularization chosen by cross-validation | 1.42 (test) |
| [task3.ipynb](src/task3.ipynb) | Polynomial regression with physics-inspired features and a search over feature groups | 1.18 (local test split, feature groups 0 + 1 + 3) |
| [task4.ipynb](src/task4.ipynb) | K-nearest neighbors regression, with and without physics-inspired features | 0.89 (validation, best group combination) |

The physics-inspired features (see `add_three_body_features` in [src/utils.py](src/utils.py)) are: pairwise distances and their inverses, distance ratios,
area and internal angles of the triangle formed by the bodies, distances to the center of mass, and an approximate angular momentum.

The RMSE values above are the ones saved in the notebook outputs and come from the local train/validation/test split, not from the Kaggle leaderboard.

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

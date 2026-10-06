"""Shared helpers for the three-body problem notebooks.

The training data is a sequence of trajectories stored back to back: every
trajectory has ``TRAJECTORY_LENGTH`` rows (one per time step) and starts at
``t == 0``.
"""

import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, MultiTaskLassoCV, RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from tqdm import tqdm

# Number of time steps in one (collision-free) trajectory.
TRAJECTORY_LENGTH = 257

# Position columns of the three bodies, in the order used for the targets.
POSITION_COLS = ["x_1", "y_1", "x_2", "y_2", "x_3", "y_3"]


# ---------------------------------------------------------------------------
# Data inspection
# ---------------------------------------------------------------------------

def dataset_summary(df):
    """Print summary statistics and plot boxplots and histograms of ``df``.

    Coordinate columns (``x_*``, ``y_*``) and velocity columns (``v_x*``,
    ``v_y*``) are handled as two separate groups.
    """
    # Dimensions
    print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns\n")

    # Basic statistics
    print("Basic statistics per column:")
    desc = df.describe().T
    print(desc[['min', 'mean', 'max']])

    # Standard deviation
    desc['std'] = df.std()
    print("\nStandard deviation:")
    print(desc['std'])

    # Split the columns into coordinates and velocities
    coord_cols = [col for col in df.columns if col.startswith(('x_', 'y_'))]
    vel_cols = [col for col in df.columns if col.startswith(('v_x', 'v_y'))]

    # Boxplots of the absolute values on a log scale
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    data_coord = [np.abs(df[c]) + 1e-8 for c in coord_cols]
    axes[0].boxplot(data_coord, tick_labels=coord_cols)
    axes[0].set_yscale('log')
    axes[0].set_title("Coordinates (x, y) - log scale (abs values)")
    axes[0].set_xlabel("Variables")
    axes[0].set_ylabel("Values (log scale)")

    data_vel = [np.abs(df[c]) + 1e-8 for c in vel_cols]
    axes[1].boxplot(data_vel, tick_labels=vel_cols)
    axes[1].set_yscale('log')
    axes[1].set_title("Velocities (v_x, v_y) - log scale (abs values)")
    axes[1].set_xlabel("Variables")
    axes[1].set_ylabel("Values (log scale)")

    plt.tight_layout()
    plt.show()

    # Raw histograms (no log scale)
    def plot_raw_hist(df, cols, color, group_name):
        for col in cols:
            plt.figure(figsize=(6, 3))
            plt.hist(df[col], bins=50, color=color, edgecolor='black')
            plt.title(f"{col} - Raw distribution ({group_name})")
            plt.xlabel("Value")
            plt.ylabel("Frequency")

            # Annotate the min and max values
            min_val = df[col].min()
            max_val = df[col].max()
            plt.annotate(f"min: {min_val:.2e}", xy=(0.98, 0.95), xycoords='axes fraction',
                         ha='right', va='top', fontsize=9, color='red')
            plt.annotate(f"max: {max_val:.2e}", xy=(0.98, 0.88), xycoords='axes fraction',
                         ha='right', va='top', fontsize=9, color='red')

            plt.tight_layout()
            plt.show()

    plot_raw_hist(df, coord_cols, color='skyblue', group_name='Coordinates')
    plot_raw_hist(df, vel_cols, color='lightcoral', group_name='Velocities')


def check_dataset_anomalies(df):
    """Report missing values, infinite values and empty columns in ``df``."""
    print("Dataset anomalies check\n")

    # 1. Missing values (NaN)
    missing = df.isnull().sum().sum()
    if missing == 0:
        print("No missing values (NaN) found.")
    else:
        print(f"Found {missing} missing values (NaN).")
        print(df.isnull().sum()[df.isnull().sum() > 0])

    # 2. Infinite values
    inf_mask = df.isin([np.inf, -np.inf])
    inf_count = inf_mask.sum().sum()
    if inf_count == 0:
        print("No infinite values (+inf, -inf) found.")
    else:
        print(f"Found {inf_count} infinite values.")
        print(inf_mask.sum()[inf_mask.sum() > 0])

    # 3. Empty columns
    empty_cols = [col for col in df.columns if df[col].isnull().all()]
    if not empty_cols:
        print("No empty columns.")
    else:
        print(f"Found empty columns: {empty_cols}")


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def clean_data(df, traj_len=TRAJECTORY_LENGTH, tol=1e-8):
    """Clean a dataset of trajectories.

    Steps:
    1. Drop the trajectories whose first row is entirely zero.
    2. Remove collisions: truncate each trajectory at the first row where all
       the features are close to zero.

    Args:
        df (pd.DataFrame): Dataset to clean (last column is the ``Id``).
        traj_len (int): Number of rows in one trajectory.
        tol (float): Tolerance under which a value is considered to be zero
            when detecting collisions.

    Returns:
        pd.DataFrame: The cleaned dataset.
    """
    # 1. Drop the Id column if present
    df_no_id = df.iloc[:, :-1] if df.shape[1] > traj_len else df.copy()

    # 2. Find the trajectories that start with a row of zeros
    num_traj = len(df_no_id) // traj_len
    zero_traj = []

    for i in range(num_traj):
        start_idx = i * traj_len
        first_row = df_no_id.iloc[start_idx]
        if (first_row == 0).all():
            zero_traj.append(i)

    # 3. Remove those trajectories
    drop_indices = []
    for traj_id in zero_traj:
        start = traj_id * traj_len
        end = (traj_id + 1) * traj_len
        drop_indices.extend(range(start, end))

    df_cleaned = df.drop(drop_indices).reset_index(drop=True)

    # 4. Remove collisions
    cleaned = []
    num_traj = len(df_cleaned) // traj_len

    for i in range(num_traj):
        start = i * traj_len
        end = (i + 1) * traj_len
        traj = df_cleaned.iloc[start:end]

        traj_features = traj.iloc[:, :-1]  # every column except the last one
        zero_mask = (np.abs(traj_features.values) < tol).all(axis=1)

        if zero_mask.any():
            first_zero = zero_mask.argmax()
            traj = traj.iloc[:first_zero]  # keep rows before the first zero row

        cleaned.append(traj)

    cleaned_df = pd.concat(cleaned).reset_index(drop=True)

    return cleaned_df


def split_trajectories(df,
                       train_size=0.6,
                       validation_size=0.2,
                       test_size=0.2,
                       method="random",
                       random_state=None):
    """Split whole trajectories into train, validation and test sets.

    A trajectory is never cut: each one starts at ``t == 0`` and goes to
    either the train, the validation or the test set. Trajectories left over
    by the integer rounding are added to the train set.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: train, validation and
        test sets.
    """
    if random_state is not None:
        np.random.seed(random_state)

    # Index of the first row of every trajectory
    traj_start_indices = df.index[df['t'] == 0].tolist()
    traj_start_indices.append(len(df))  # end of the last trajectory

    # (start, end) slice of every trajectory
    traj_slices = [(traj_start_indices[i], traj_start_indices[i + 1])
                   for i in range(len(traj_start_indices) - 1)]

    if method == "random":
        np.random.shuffle(traj_slices)
    else:
        raise NotImplementedError(f"Method {method} is not implemented yet")

    n_traj = len(traj_slices)
    n_train = int(train_size * n_traj)
    n_val = int(validation_size * n_traj)
    n_test = int(test_size * n_traj)

    # Give the remaining trajectories to the train set
    n_remaining = n_traj - (n_train + n_val + n_test)
    n_train += n_remaining

    train_slices = traj_slices[:n_train]
    val_slices = traj_slices[n_train:n_train + n_val]
    test_slices = traj_slices[n_train + n_val:n_train + n_val + n_test]

    # Concatenate the rows of the selected trajectories
    train_df = pd.concat([df.iloc[start:end] for start, end in train_slices])
    val_df = pd.concat([df.iloc[start:end] for start, end in val_slices])
    test_df = pd.concat([df.iloc[start:end] for start, end in test_slices])

    return train_df, val_df, test_df


def replicate_initial_position_by_block(df):
    """Copy the initial positions of each trajectory onto all of its rows.

    The models predict the position at time ``t`` from the initial positions
    only, so every row of a trajectory gets the positions of its ``t == 0`` row.
    """
    copy = df.copy()
    data = copy[POSITION_COLS].values
    t_values = copy["t"].values

    # Start of each trajectory block
    block_starts = np.where(t_values == 0)[0]
    block_starts = np.append(block_starts, len(df))  # end of the last block

    # Replicate the initial position inside each block
    for i in range(len(block_starts) - 1):
        start, end = block_starts[i], block_starts[i + 1]
        data[start:end] = data[start]

    copy[POSITION_COLS] = data
    return copy


def get_n_trajectories(df, n):
    """Return the first ``n`` trajectories of ``df`` (all of them if fewer)."""
    start_indices = df.index[df['t'] == 0].tolist()
    trajectory_blocks = []

    for i in range(min(n, len(start_indices))):
        start_idx = start_indices[i]
        end_idx = start_indices[i + 1] if i + 1 < len(start_indices) else None

        # Convert index labels to row positions
        start_pos = df.index.get_loc(start_idx)
        end_pos = df.index.get_loc(end_idx) if end_idx else None

        trajectory_blocks.append(df.iloc[start_pos:end_pos])

    return pd.concat(trajectory_blocks)


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def add_three_body_features(df, masses=(1, 1, 1), G=1.0):
    """Add physics-inspired features computed from the three body positions.

    New columns: pairwise distances (``r_ij``) and their inverses, distance
    ratios, triangle area, internal triangle angles, distances to the center
    of mass (``d*_cm``) and an approximate angular momentum (``Lz``).

    Args:
        df (pd.DataFrame): Must contain ``x_i`` and ``y_i`` for i in 1..3.
        masses (tuple): Masses of the three bodies.
        G (float): Gravitational constant (currently unused).

    Returns:
        pd.DataFrame: A copy of ``df`` with the extra columns.
    """
    df_new = df.copy()
    eps = 1e-8  # avoids divisions by zero
    m1, m2, m3 = masses

    # Positions
    r1 = df[['x_1', 'y_1']].values
    r2 = df[['x_2', 'y_2']].values
    r3 = df[['x_3', 'y_3']].values

    # Pairwise distances
    r12 = np.linalg.norm(r1 - r2, axis=1)
    r13 = np.linalg.norm(r1 - r3, axis=1)
    r23 = np.linalg.norm(r2 - r3, axis=1)

    df_new['r_12'] = r12
    df_new['r_13'] = r13
    df_new['r_23'] = r23

    # Inverse distances
    df_new['inv_r_12'] = 1.0 / (r12 + eps)
    df_new['inv_r_13'] = 1.0 / (r13 + eps)
    df_new['inv_r_23'] = 1.0 / (r23 + eps)

    # Distance ratios
    df_new['r12_over_r13'] = r12 / (r13 + eps)
    df_new['r12_over_r23'] = r12 / (r23 + eps)
    df_new['r13_over_r23'] = r13 / (r23 + eps)

    # Area of the triangle formed by the three bodies
    df_new['triangle_area'] = 0.5 * np.abs(
        (r2[:, 0] - r1[:, 0]) * (r3[:, 1] - r1[:, 1])
        - (r3[:, 0] - r1[:, 0]) * (r2[:, 1] - r1[:, 1])
    )

    # Internal angles of the triangle
    def angle(a, b, c):
        # Law of cosines
        cos_angle = (b**2 + c**2 - a**2) / (2 * b * c + eps)
        return np.arccos(np.clip(cos_angle, -1, 1))

    df_new['angle_1'] = angle(r23, r12, r13)
    df_new['angle_2'] = angle(r13, r12, r23)
    df_new['angle_3'] = angle(r12, r13, r23)

    # Center of mass and distance of each body to it
    X_cm = (m1 * r1[:, 0] + m2 * r2[:, 0] + m3 * r3[:, 0]) / (m1 + m2 + m3)
    Y_cm = (m1 * r1[:, 1] + m2 * r2[:, 1] + m3 * r3[:, 1]) / (m1 + m2 + m3)

    df_new['d1_cm'] = np.linalg.norm(r1 - np.stack([X_cm, Y_cm], axis=1), axis=1)
    df_new['d2_cm'] = np.linalg.norm(r2 - np.stack([X_cm, Y_cm], axis=1), axis=1)
    df_new['d3_cm'] = np.linalg.norm(r3 - np.stack([X_cm, Y_cm], axis=1), axis=1)

    # Approximate angular momentum around the center of mass.
    # Lz = sum_i m_i * ((x_i - X_cm) * y_i - (y_i - Y_cm) * x_i), simplified for 2D.
    Lz = m1 * ((r1[:, 0] - X_cm) * r1[:, 1] - (r1[:, 1] - Y_cm) * r1[:, 0]) + \
         m2 * ((r2[:, 0] - X_cm) * r2[:, 1] - (r2[:, 1] - Y_cm) * r2[:, 0]) + \
         m3 * ((r3[:, 0] - X_cm) * r3[:, 1] - (r3[:, 1] - Y_cm) * r3[:, 0])
    df_new['Lz'] = Lz

    return df_new


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def validate_poly_regression(X_train, y_train, X_val, y_val, regressor=None,
                             degrees=range(2, 3), max_features=None):
    """Fit polynomial regressions and keep the one with the lowest validation RMSE.

    Args:
        regressor (str | None): ``'L2'`` for ridge regression (``RidgeCV``),
            ``'L1'`` for lasso (``MultiTaskLassoCV``), anything else for plain
            linear regression. The regularization strength is chosen by
            5-fold cross-validation over the same grid of alphas.
        degrees (iterable): Polynomial degrees to try.
        max_features (int | None): Skip the degrees that would produce more
            polynomial features than this.

    Returns:
        tuple: ``(best_model, best_rmse, best_degree, best_reg, best_alpha)``.
    """
    start_time = time.time()

    best_rmse = np.inf
    best_model = None
    best_degree = None
    best_alpha = None
    best_reg = None

    # Same grid of alpha values for L1 and L2
    alphas = np.logspace(-5, 5, 13)

    for degree in tqdm(degrees, desc="Testing degrees"):
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        poly.fit(X_train)
        n_features = poly.n_output_features_
        print(f"Degree {degree}: {n_features} features")

        if max_features and n_features > max_features:
            print(f" Skipped degree {degree} (too many features: {n_features})")
            continue

        if regressor == 'L2':
            model = RidgeCV(
                alphas=alphas,
                scoring='neg_mean_squared_error',
                cv=5
            )
            reg_name = 'L2'
        elif regressor == 'L1':
            model = MultiTaskLassoCV(
                alphas=alphas,
                cv=5
            )
            reg_name = 'L1'
        else:
            # Plain linear regression
            model = LinearRegression()
            reg_name = 'None'

        pipe = Pipeline([
            ('poly', poly),
            ('scaler', StandardScaler()),
            ('reg', model)
        ])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))

        if rmse < best_rmse:
            best_rmse = rmse
            best_model = pipe
            best_degree = degree
            best_alpha = pipe['reg'].alpha_ if reg_name != 'None' else None
            best_reg = reg_name

    elapsed_time = time.time() - start_time
    print(f"\n⏱️ Total elapsed time: {elapsed_time:.2f} seconds")
    print(f"✅ Best model: degree={best_degree}, RMSE={best_rmse:.4f}, reg={best_reg}, alpha={best_alpha}")

    return best_model, best_rmse, best_degree, best_reg, best_alpha


def validate_knn_regression(X_train, y_train, X_val, y_val, k_values=range(1, 15)):
    """Validate a KNN regressor for a range of ``k`` values.

    Scales the features and the targets, computes the validation RMSE,
    records the training and inference times, and plots them against ``k``.

    Returns:
        pd.DataFrame: One row per ``k`` with ``rmse``, ``train_time`` and
        ``inference_time``.
    """
    results = []

    # Scale features and targets
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train_scaled = scaler_X.fit_transform(X_train)
    X_val_scaled = scaler_X.transform(X_val)

    y_train_scaled = scaler_y.fit_transform(y_train)

    for k in k_values:
        knn = KNeighborsRegressor(n_neighbors=k, weights='distance')

        # Training time
        start_train = time.time()
        knn.fit(X_train_scaled, y_train_scaled)
        end_train = time.time()
        train_time = end_train - start_train

        # Inference time
        start_inf = time.time()
        y_val_pred_scaled = knn.predict(X_val_scaled)
        end_inf = time.time()
        inference_time = end_inf - start_inf

        # Back to the original scale
        y_val_pred = scaler_y.inverse_transform(y_val_pred_scaled)

        rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))

        results.append({
            'k': k,
            'rmse': rmse,
            'train_time': train_time,
            'inference_time': inference_time
        })

        print(f"k = {k}, RMSE = {rmse:.5f}, Train time = {train_time:.3f}s, Inference time = {inference_time:.3f}s")

    results_df = pd.DataFrame(results)

    # RMSE vs k
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(results_df['k'], results_df['rmse'], marker='o')
    plt.xlabel("k")
    plt.ylabel("Validation RMSE")
    plt.title("KNN Regression: RMSE vs k")
    plt.grid(True)

    # Training and inference times vs k
    plt.subplot(1, 2, 2)
    plt.plot(results_df['k'], results_df['train_time'], marker='o', label='Training time')
    plt.plot(results_df['k'], results_df['inference_time'], marker='o', label='Inference time')
    plt.xlabel("k")
    plt.ylabel("Time (seconds)")
    plt.title("Training and Inference Time vs k")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    return results_df


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_y_yhat(y_test, y_pred, plot_title="plot"):
    """Scatter plot of true vs. predicted positions, one subplot per coordinate.

    At most 500 randomly chosen points are plotted. The red line is the
    ideal ``y_pred == y_test`` diagonal. ``plot_title`` is currently unused.
    """
    labels = POSITION_COLS
    MAX = 500
    if len(y_test) > MAX:
        idx = np.random.choice(len(y_test), MAX, replace=False)
    else:
        idx = np.arange(len(y_test))
    plt.figure(figsize=(10, 10))
    for i in range(6):
        x0 = np.min(y_test[idx, i])
        x1 = np.max(y_test[idx, i])
        plt.subplot(3, 2, i + 1)
        plt.scatter(y_test[idx, i], y_pred[idx, i])
        plt.xlabel('True ' + labels[i])
        plt.ylabel('Predicted ' + labels[i])
        plt.plot([x0, x1], [x0, x1], color='red')
        plt.axis('square')
    plt.show()

import calendar
import gpytorch
import numpy as np
import pandas as pd
import time
import torch
from sklearn.model_selection import KFold
from tqdm import tqdm


def load_and_merge_weather_data(paths):
    """Loads, processes, and merges multiple weather data CSV files.

    Each file is expected to contain hourly weather measurements at different
    geographic locations. The expected date format is YYYY-MM-DD. The function:

    - Parses the 'time' column to timezone-aware datetime format (UTC).
    - Drops location and forecast-specific columns: 'forecast_origin', 'longitude', 'latitude'.
    - Aggregates data by averaging all features across locations for each timestamp.
    - Concatenates all datasets into a single time-ordered DataFrame.
    - Performs a sanity check to ensure no data was lost during concatenation.

    Args:
        paths (list[str]): List of file paths to weather CSV files.

    Returns:
        pd.DataFrame: Time-indexed DataFrame with averaged weather variables.

    Raises:
        AssertionError: If the number of rows after merging doesn't match the expected count.
    """
    dfs = []
    for path in paths:
        df = pd.read_csv(path, sep=',', decimal='.')
        df['time'] = pd.to_datetime(df['time'], utc=True)  # transforms date columns to the format: YYYY-MM-DD HH:MM:SS
        df = df.drop(columns=['forecast_origin', 'longitude', 'latitude'])
        # Average values column-wise as there are data points from different positions (long/lat) for each hour
        df = df.groupby('time', as_index=False).mean(numeric_only=True)
        dfs.append(df)

    combined_df = pd.concat(dfs, ignore_index=True).sort_values('time')

    # Sanity check: ensure no rows were lost
    expected_length = sum(len(df) for df in dfs)
    assert len(combined_df) == expected_length, (
        f"Row mismatch after merge: expected {expected_length}, got {len(combined_df)}"
    )

    return combined_df


def parse_german_float(x: str) -> float:
    """Convert German-formatted number to float."""
    return float(x.replace('.', '').replace(',', '.'))


def load_supply_data(path, column='Solar Power [MW]'):
    """ Loads and aggregates solar power supply data from a CSV file. The expected date format is DD-MM-YY.

    This function:
    - Reads the 'Date from' and 'Solar Power [MW]' columns from a semicolon-delimited CSV file.
    - Converts German-style decimal commas to periods for proper float parsing.
    - Converts timestamps in 'Date from' to timezone-aware datetime objects (UTC).
    - Floors timestamps to the nearest hour.
    - Aggregates 15-minute interval data to hourly resolution by computing both the sum (MWh)
      and mean (MW) of solar power per hour.

    The returned DataFrame contains:
        - 'time': hourly timestamps (UTC)
        - 'solar_supply_sum': total solar energy per hour (MWh)
        - 'solar_supply_mean': average solar power (MW) over each hour

    Args:
        path (str): File path to the CSV supply data.

    Returns:
        pd.DataFrame: Hourly solar supply data with columns ['time', 'solar_supply_sum', 'solar_supply_mean']
    """
    df = pd.read_csv(path, sep=';', decimal=',',
                     usecols=['Date from', column],
                     converters={'Solar Power [MW]': parse_german_float})
    df['time'] = pd.to_datetime(df['Date from'], format='%d.%m.%y %H:%M', utc=True)  # convert to date format
    # Floor times to the hour -> 13:00, 13:15, 13:30, 13:45 -> all turn into 13:00
    df['time'] = df['time'].dt.floor('h')
    # Create columns for sum and mean by aggregating the power generation values of each hour
    df = (df.groupby('time')
          .agg(solar_supply_sum=('Solar Power [MW]', 'sum'), solar_supply_mean=('Solar Power [MW]', 'mean'))
          .reset_index())
    return df


def load_installed_cap_data(path):
    """Loads photovoltaic capacity data and expands it to hourly resolution. The expected date format is DD-MM-YY.

    Each row specifies the installed capacity valid from 'Date from' to 'Date to'.
    The returned series provides hourly installed capacity values over the full time range.

    Args:
        path (str): Path to the installed capacity CSV file.

    Returns:
        pd.Series: A time-indexed Series with hourly resolution of installed capacity in MW.
    """
    df = pd.read_csv(path, sep=';', decimal=',',
                     usecols=['Date from', 'Date to', 'Photovoltaic [MW]'],
                     converters={'Photovoltaic [MW]': parse_german_float})
    df['Date from'] = pd.to_datetime(df['Date from'], format='%d.%m.%y', utc=True)
    df['Date to'] = pd.to_datetime(df['Date to'], format='%d.%m.%y', utc=True)

    # Hourly grid from earliest 'Date from' to latest 'Date to' entry. This is alright as the intersection of all tables would cut out the last year anyway
    full_hours = pd.date_range(start=df['Date from'].min(), end=df['Date to'].max(), freq='h')
    # Create a series indexed by 'Date from'
    cap_series = df.set_index('Date from')['Photovoltaic [MW]'].sort_index()
    # Reindex to hourly and forward-fill missing values
    cap_hourly = (
        cap_series
        .reindex(full_hours)
        .interpolate(method='time')  # linear interpolation between known points
        # .ffill()  # carry the last known value through to the final timestamp
    )
    return cap_hourly


def get_time_intersection(weather_df, supply_hourly, capacity_df):
    t0 = max(
        weather_df['time'].min(),
        supply_hourly['time'].min(),
        capacity_df.index.min()
    )
    t1 = min(
        weather_df['time'].max(),
        supply_hourly['time'].max(),
        capacity_df.index.max()
    )
    print(f'Common window: {t0} - {t1}')

    common_time_idx = pd.date_range(start=t0, end=t1, freq='h', tz='UTC')
    return common_time_idx


def concat_tables(common_time_idx, supply_hourly, weather_df, capacity_df, target_col='solar_supply_sum'):
    """Concatenates and aligns time series data from solar supply, weather, and installed capacity tables.

    Aligns each input DataFrame (`supply_hourly`, `weather_df`, and `capacity_df`) to a common datetime index.
    It interpolates missing values using time-based interpolation and then merges the aligned tables into a single DataFrame.
    It also renames the target metric (sum or mean) to 'solar_supply' and drops the other.

    Args:
        common_time_idx (pd.DatetimeIndex): The common datetime index to which all tables will be aligned.
        supply_hourly (pd.DataFrame): DataFrame containing hourly solar supply data with a 'time' column
            and 'solar_supply_sum' and 'solar_supply_mean' columns.
        weather_df (pd.DataFrame): DataFrame containing weather features with a 'time' column.
        capacity_df (pd.Series or pd.DataFrame): DataFrame representing installed capacity
            indexed by time or with a 'time' column.
        target_col (str, optional): Specifies which solar supply metric to use as the target.
            Must be either 'solar_supply_sum' or 'solar_supply_mean'. Defaults to 'solar_supply_sum'.

    Returns:
        pd.DataFrame: A single DataFrame with aligned and merged data including weather,
        capacity, and selected solar supply values under the column name 'solar_supply'.

    Raises:
        ValueError: If `target_col` is not 'solar_supply_sum' or 'solar_supply_mean'.
    """
    # Align each table on the common_time_idx
    weather_aligned = (
        weather_df
        .set_index('time')
        .reindex(common_time_idx)
        .interpolate(method='time')
        .ffill()
    )

    supply_aligned = (
        supply_hourly
        .set_index('time')
        .reindex(common_time_idx)
        .interpolate(method='time')
    )

    capacity_aligned = (
        capacity_df
        .reindex(common_time_idx)
        .interpolate(method='time')
    )

    # Merge them into one DataFrame
    full_data = pd.concat([
        supply_aligned,
        weather_aligned,
        capacity_aligned.rename('Installed_Capacity')
    ], axis=1)

    # Sanity‐check
    # print('Any NaNs left? ', full_data.isna().any().any())

    # Chose the target column. This is either the sum or mean of the solar supply per hour. Drop the other one
    drop_target = 'solar_supply_mean' if target_col == 'solar_supply_sum' else 'solar_supply_sum'
    full_data.drop(columns=[drop_target], inplace=True)
    full_data.rename(columns={target_col: 'solar_supply'}, inplace=True)

    return full_data


def add_hour_column(df):
    """
    Adds sinusoidal time columns encoding the hour of day to a DataFrame which is indexed by time.
    Changes are done in-place

    Args:
        df (pd.DataFrame): Dataframe with time-indexed rows
    Returns:
        None
    """
    hour_norm = 2 * np.pi * df.index.hour / 24
    df['hour_sin'] = np.sin(hour_norm)
    df['hour_cos'] = np.cos(hour_norm)


def add_year_column(df):
    """
    Adds sinusoidal time columns encoding the year to a DataFrame which is indexed by time.
    Changes are done in-place

    Args:
        df (pd.DataFrame): Dataframe with time-indexed rows
    Returns:
        None
    """
    years = df.index.year
    days_in_year = np.where([calendar.isleap(y) for y in years], 366,
                            365)  # this accounts the sine/cos wave for leap years
    day_of_year_ang = 2 * np.pi * (df.index.dayofyear - 1) / days_in_year
    df['day_of_year_sin'] = np.sin(day_of_year_ang)
    df['day_of_year_cos'] = np.cos(day_of_year_ang)


def df_to_tensor(df, device):
    """Convert a pd.DataFrame to a torch float tensor on `device`."""
    return torch.from_numpy(df.values).float().to(device)


def train_and_eval(X_tr, y_tr, X_val, y_val,
                   kernel_builder, mean_builder,
                   num_inducing, num_latents, lr, num_epochs, GPModel, device, fold_i=None, outer_pbar=None):
    """
    Trains a variational Gaussian Process model using GPyTorch and evaluates it on a validation set.

    This function builds a batched variational GP model with the specified kernel and mean functions,
    trains it using the ELBO objective, and computes the RMSE on the validation set.

    Args:
        X_tr (torch.Tensor): Training input features of shape (n_train, input_dim).
        y_tr (torch.Tensor): Training targets of shape (n_train, output_dim).
        X_val (torch.Tensor): Validation input features of shape (n_val, input_dim).
        y_val (torch.Tensor): Validation targets of shape (n_val, output_dim).
        kernel_builder (Callable[[int], gpytorch.kernels.Kernel]): Function that returns a batched kernel module.
        mean_builder (Callable[[int], gpytorch.means.Mean]): Function that returns a batched mean module.
        num_inducing (int): Number of inducing points for the variational GP.
        num_latents (int): Number of latent GPs in the variational model.
        lr (float): Learning rate for the Adam optimizer.
        num_epochs (int): Number of training epochs.
        device (torch.device): Device to run the training on (e.g., 'cuda' or 'cpu').

    Returns:
        float: Root Mean Squared Error (RMSE) of the model predictions on the validation set.
    """
    # slice inducing points
    inducing = X_tr[:num_inducing].clone()

    # build mean & covar modules (batched over num_latents)
    mean_mod = mean_builder(num_latents).to(device)
    covar_mod = kernel_builder(num_latents).to(device)

    # instantiate model + likelihood
    model = GPModel(inducing, y_tr.shape[1], num_latents, mean_mod, covar_mod).to(device)
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=y_tr.shape[1]).to(device)

    # optim + ELBO
    optimizer = torch.optim.Adam([
        {'params': model.parameters()},
        {'params': likelihood.parameters()}
    ], lr=lr)
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=X_tr.size(0))

    model.train()
    likelihood.train()

    train_ds = torch.utils.data.TensorDataset(X_tr, y_tr)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=512, shuffle=True)
    for epoch in range(num_epochs):
        epoch_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            output = model(X_batch)
            loss = -mll(output, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if outer_pbar is not None:
            outer_pbar.set_description(f"Fold {fold_i + 1} | Epoch {epoch + 1}/{num_epochs}")
            outer_pbar.refresh()

    # eval
    model.eval();
    likelihood.eval()
    with torch.no_grad():
        pred = likelihood(model(X_val)).mean.cpu()
    rmse = torch.sqrt(((pred - y_val.cpu()) ** 2).mean()).item()
    return rmse


def perform_gridsearch(kernel_builders, mean_builders, inducing_list, latents_list, epochs_list, lr_list, device,
                       RANDOM_SEED, X_train, y_train, GPModel):
    kf = KFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED)
    best = {'rmse': float('inf')}

    total_it = len(kernel_builders) * len(mean_builders) * len(inducing_list) * len(latents_list) * len(lr_list) * len(
        epochs_list)
    curr_iteration = 1
    # Grid Search through all parameters
    for kernel_name, kbuilder in kernel_builders.items():
        for mean_name, mbuilder in mean_builders.items():
            for num_inducing in inducing_list:
                for num_latents in latents_list:
                    for lr in lr_list:
                        for epochs in epochs_list:
                            start_time = time.time()
                            rmses = []
                            folds = list(kf.split(X_train))
                            fold_pbar = tqdm(total=len(folds),
                                             desc=f"[{curr_iteration}/{total_it}] {kernel_name}-{mean_name}",
                                             position=0)
                            for fold_i, (tr_idx, val_idx) in enumerate(folds):
                                X_tr, X_val = X_train[tr_idx], X_train[val_idx]
                                y_tr, y_val = y_train[tr_idx], y_train[val_idx]
                                rmse = train_and_eval(
                                    X_tr, y_tr, X_val, y_val,
                                    kbuilder, mbuilder,
                                    num_inducing, num_latents,
                                    lr, epochs, GPModel, device, fold_i=fold_i, outer_pbar=fold_pbar
                                )
                                fold_pbar.set_postfix(rmse=rmse)
                                rmses.append(rmse)
                            fold_pbar.close()
                            # Average rmse across folds
                            avg_rmse = np.mean(rmses)
                            print(
                                f'{curr_iteration:3d}/{total_it} in {time.time() - start_time:5.2f}s | GP({mean_name}, {kernel_name}) | '
                                f'num_inducing={num_inducing}, num_latents={num_latents}, '
                                f'lr={lr:.4f}, epochs={epochs} → '
                                f'RMSE={avg_rmse:.4f}')
                            curr_iteration += 1
                            if avg_rmse < best['rmse']:
                                best.update({
                                    'rmse': avg_rmse,
                                    'kernel': kernel_name,
                                    'mean': mean_name,
                                    'num_inducing': num_inducing,
                                    'num_latents': num_latents,
                                    'lr': lr,
                                    'epochs': epochs
                                })
    return best


def save_data(X_train, X_test, y_train, y_test, dates_train, dates_test, fpath):
    d = {'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test, 'dates_train': dates_train,
         'dates_test': dates_test}
    torch.save(d, fpath)
    print("Datasets saved successfully")

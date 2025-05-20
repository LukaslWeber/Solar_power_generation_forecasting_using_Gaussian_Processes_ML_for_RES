import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
        df['time'] = pd.to_datetime(df['time'], utc=True) # transforms date columns to the format: YYYY-MM-DD HH:MM:SS
        df = df.drop(columns=['forecast_origin', 'longitude', 'latitude'])
        # Average values column-wise as there are data points from different positions (long/lat) for each hour
        # TODO: Maybe instead of averaging, use a PCA to weight the different locations for each hour
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
        capacity_df['Date from'].min()
    )
    t1 = min(
        weather_df['time'].max(),
        supply_hourly['time'].max(),
        capacity_df['Date to'].max()
    )
    print(f'Common window: {t0} - {t1}')

    common_time_idx = pd.date_range(start=t0, end=t1, freq='h', tz='UTC')
    return common_time_idx
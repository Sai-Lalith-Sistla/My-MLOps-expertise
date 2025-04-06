import pandas as pd



def handle_missing_values(df):
        missing_report = df.isnull().mean().compute()
        df = df.ffill().bfill()
        return df, missing_report


# Creating lag features
def create_lag_features(df, lags=[1, 7, 28]):
    def add_lag_features(df, lags=[1, 7, 28]):
        for lag in lags:
            df[f'lag_{lag}'] = df.groupby('id')['sales'].shift(lag)
        return df

    # Create updated meta
    new_cols = {f'lag_{lag}': 'float64' for lag in [1, 7, 28]}
    meta = df._meta.assign(**{k: pd.Series(dtype=v) for k,v in new_cols.items()})
    df = df.map_partitions(add_lag_features, meta=meta)
    return df



# Rolling window features
def create_rolling_features(df, window_sizes=[7, 14]):
    def rolling_func(partition_df):
        partition_df = partition_df.sort_values(['id', 'day'])
        for window in window_sizes:
            partition_df[f'rolling_mean_{window}'] = (
                partition_df.groupby('id')['sales']
                            .rolling(window=window, min_periods=1)
                            .mean()
                            .reset_index(drop=True)
            )
        return partition_df
    
    # Create updated meta
    new_cols = {f'rolling_mean_{w}': 'float64' for w in window_sizes}
    meta = df._meta.assign(**{k: pd.Series(dtype=v) for k,v in new_cols.items()})
    df = df.map_partitions(rolling_func, meta=meta)
    return df

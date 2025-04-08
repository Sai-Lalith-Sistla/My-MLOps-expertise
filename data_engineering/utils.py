import pandas as pd
import dask.dataframe as dd
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder


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
    new_cols = {f'lag_{lag}': 'int32' for lag in [1, 7, 28]}
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
    new_cols = {f'rolling_mean_{w}': 'int32' for w in window_sizes}
    meta = df._meta.assign(**{k: pd.Series(dtype=v) for k,v in new_cols.items()})
    df = df.map_partitions(rolling_func, meta=meta)
    return df



class DaskLabelEncoderManager:
    """
    Manages label encoding for multiple columns in a Dask DataFrame.
    Safely handles unseen categories (-1) and allows saving/loading encoders.
    """

    def __init__(self, columns=None):
        self.columns = columns or []
        self.encoders = {}

    def _fit_label_encoder(self, values):
        le = LabelEncoder()
        le.fit(values.astype(str))
        return le

    def _safe_transform(self, encoder, values):
        values = np.array(values).astype(str)
        known_classes = set(encoder.classes_)
        transformed = np.full(values.shape, -1, dtype=int)  # Default -1 for unknown

        mask_known = np.isin(values, list(known_classes))
        if mask_known.any():
            transformed[mask_known] = encoder.transform(values[mask_known])
        return transformed

    def fit(self, df):
        """
        Fit label encoders on specified columns of a Dask DataFrame.
        """
        df_local = df[self.columns].compute()

        for col in self.columns:
            le = self._fit_label_encoder(df_local[col])
            self.encoders[col] = le

    def transform(self, df):
        """
        Transform specified columns of a Dask DataFrame using fitted encoders.
        Returns:
            dask.dataframe.DataFrame: Transformed DataFrame.
        """
        if not self.encoders:
            raise ValueError("Encoders are not fitted. Call fit() first.")
        
        def encode_partition(partition, encoders):
            for col, le in encoders.items():
                partition[col] = self._safe_transform(le, partition[col]).astype('int32')
            return partition

        # # Build new meta manually
        # meta_dict = {}
        # for col in self.columns:
        #     meta_dict[col] = 'int32'
        # new_meta = df._meta.assign(**{k: pd.Series(dtype=v) for k,v in meta_dict.items()})
        # print(type(new_meta), new_meta)
        
        return df.map_partitions(encode_partition, encoders=self.encoders) #, meta=new_meta


    def fit_transform(self, df):
        """
        Fit encoders and transform the DataFrame.
        """
        self.fit(df)
        return self.transform(df)

    def save(self, filepath):
        """
        Save encoders and columns to a pickle file.
        """
        with open(filepath, 'wb') as f:
            pickle.dump({
                'columns': self.columns,
                'encoders': self.encoders
            }, f)

    def load(self, filepath):
        """
        Load encoders and columns from a pickle file.
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.columns = data['columns']
            self.encoders = data['encoders']

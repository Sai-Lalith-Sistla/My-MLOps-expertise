import pandas as pd
import numpy as np

def preprocess_input(raw_json):
    """
    Preprocess user input JSON and return a clean numpy array.
    """
    # Example: Assume input is {"feature1": 123, "feature2": 45.6, ...}
    try:
        df = pd.DataFrame([raw_json])  # Convert dict to DataFrame
        #####
        #  Pre processing happens here
        #####
        return df.values
    except Exception as e:
        raise Exception(f"Error in preprocessing: {e}")

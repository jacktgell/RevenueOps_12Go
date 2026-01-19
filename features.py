import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

class FeatureEngineering:
    """
    Static utility class for transforming clean EDA data into
    Machine Learning-ready sets.
    """

    @staticmethod
    def prepare_modeling_data(input_df, cat_cols, cols_to_drop):
        """
        Encapsulates the feature engineering pipeline:
        1. Extracts 'visit_day' from dates.
        2. Drops toxic leakage columns and noise.
        3. One-Hot Encodes categorical variables.
        """
        # Working on a copy to avoid modifying the original
        df = input_df.copy()

        # --- 1. Feature Extraction ---
        # We must do this BEFORE dropping 'visit_date'
        if 'visit_date' in df.columns:
            df['visit_day'] = df['visit_date'].dt.day_name()

        # --- 2. Leakage & Noise Removal ---

        df = df.drop(columns=cols_to_drop, errors='ignore')

        # --- 3. One-Hot Encoding ---

        # Check current categorical columns for display
        valid_cats = [c for c in cat_cols if c in df.columns]

        # Perform Encoding
        df_encoded = pd.get_dummies(df, columns=valid_cats, drop_first=False, dtype=int)

        return df_encoded

    @staticmethod
    def scale_numeric_features(train_df, test_df, cols_to_scale):
        """
        Fits scaler on TRAIN only, then transforms BOTH Train and Test.
        Ensures the model sees a consistent 'world view' across datasets.
        """

        train = train_df.copy()
        test = test_df.copy()

        valid_cols = [c for c in cols_to_scale if c in train.columns]

        if not valid_cols:
            print("No columns to scale.")
            return train, test

        print(f"Scaling {len(valid_cols)} numeric features...")

        scaler = RobustScaler()

        scaler.fit(train[valid_cols])

        train[valid_cols] = scaler.transform(train[valid_cols])
        test[valid_cols] = scaler.transform(test[valid_cols])

        return train, test

    @staticmethod
    def global_random_split(df, test_size=0.2, random_state=42):
        """
        Performs the ONE true split of the raw data.
        Returns full DataFrames so we can filter them later.
        """
        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            stratify=df['booked']
        )
        return train_df, test_df
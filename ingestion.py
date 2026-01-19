import pandas as pd
from typing import List

class ProjectConfig:
    """Global configuration for reproducibility and paths."""
    RANDOM_STATE: int = 42
    RAW_DATA_PATH: str = 'pricing_12go.csv'  #
    # We explicitly define garbage values mentioned in the specs
    NA_VALUES: List[str] = ['?', 'error', 'null', 'N/A', '']

class DataIngestion:
    """
    Handles raw data loading, standardizing null values, and removing exact duplicates.
    """

    def run(self, path: str = ProjectConfig.RAW_DATA_PATH) -> pd.DataFrame:
        print(f"Initiating data load from {path}...")

        try:
            df = pd.read_csv(
                path,
                na_values=ProjectConfig.NA_VALUES,
                keep_default_na=True
            )

            initial_count = len(df)
            print(f"Loaded {initial_count} rows and {df.shape[1]} columns.")

            # Enforce data uniqueness
            df = self._remove_duplicates(df)

            return df

        except FileNotFoundError:
            print(f"CRITICAL: File {path} not found.")
            raise

    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Identifies and removes duplicate rows to ensure session uniqueness.
        """
        duplicate_count = df.duplicated().sum()

        if duplicate_count > 0:
            df.drop_duplicates(inplace=True)
            print(f"Data Cleaning: Removed {duplicate_count} duplicate rows.")

        return df
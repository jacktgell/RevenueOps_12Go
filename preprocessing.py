import pandas as pd


class DataPreprocessing():
    """
    Handles data cleaning, imputation, and type casting.
    Separates 'Diagnosis' (Printing) from 'Action' (Fixing) for interview clarity.
    """

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        # Wrapper if you want to run all at once, but we will call individual steps in the notebook
        self.diagnose_dates(df)
        df = self.parse_dates(df)
        self.inspect_trip_ids(df)
        df = self.impute_trip_ids(df)
        return df

    # --- DATE HANDLING ---
    def diagnose_dates(self, df: pd.DataFrame):
        """Prints evidence of inconsistent date formats."""
        col = 'visit_date'
        print(f"--- DIAGNOSIS: {col} ---")
        print(f"Current Dtype: {df[col].dtype}")

        # Show the mix of formats (Standard ISO vs Slashes)
        print("Sample Raw Values (Note the mixed formats):")
        # We explicitly look for slash formats to highlight the issue
        slash_dates = df[df[col].astype(str).str.contains('/')][col].head(2)
        iso_dates = df[df[col].astype(str).str.contains('-')][col].head(2)

        print(pd.concat([slash_dates, iso_dates]))
        print("Issue: Mixed formats prevent chronological sorting and plotting.")

    def parse_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardizes dates and verifies the fix."""
        print("--- ACTION: Standardizing Dates ---")

        df['visit_date'] = pd.to_datetime(df['visit_date'], errors='coerce', format='mixed', dayfirst=True)

        # The Verification
        min_date, max_date = df['visit_date'].min(), df['visit_date'].max()

        # Check if we still have NaTs (Real garbage)
        nat_count = df['visit_date'].isna().sum()

        print(f"Converted to {df['visit_date'].dtype}")
        print(f"Valid Date Range: {min_date} to {max_date}")
        print(f"Remaining NaTs (Garbage): {nat_count}")
        print(f"DF Dimensions: {df.shape}")
        return df

    # --- TRIP ID HANDLING ---
    def inspect_trip_ids(self, df: pd.DataFrame):
        """Generates the Known vs Unknown report."""
        # 1. Integrity Check
        cols = ['origin', 'destination', 'transport_type']
        consistency = df.dropna(subset=['trip_id']).groupby('trip_id')[cols].nunique()

        if consistency.max().max() == 1:
            print("INTEGRITY PASSED: All trip_ids map to unique, consistent routes.")
        else:
            print("INTEGRITY FAILED: Conflicting data found.")
            return

        # 2. Known Trips
        print("--- KNOWN TRIPS (Reference) ---")
        known = df.dropna(subset=['trip_id']).groupby('trip_id').agg(
            count=('trip_id', 'size'),
            origin=('origin', 'first'),
            destination=('destination', 'first'),
            transport=('transport_type', 'first'),
        ).sort_index()
        print(known)

        # 3. Unknown Trips
        print("--- UNKNOWN TRIPS (NaNs) ---")
        unknown = df[df['trip_id'].isna()].groupby(cols).size().reset_index(name='count')
        print(unknown)

    def impute_trip_ids(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies the 1-to-1 mapping fix."""
        # Create Map
        cols = ['origin', 'destination', 'transport_type']
        route_map = df.dropna(subset=['trip_id']).set_index(cols)['trip_id'].to_dict()

        # Apply Fix
        def fill_missing(row):
            if pd.isna(row['trip_id']):
                return route_map.get((row['origin'], row['destination'], row['transport_type']), None)
            return row['trip_id']

        df['trip_id'] = df.apply(fill_missing, axis=1)

        print(f"ACTION: Imputed missing trip_ids. Remaining NaNs: {df['trip_id'].isna().sum()}")
        return df

    def verify_booking_logic(self, df: pd.DataFrame):
        """
        Validates that missing values in booking-specific columns
        align perfectly with unbooked sessions.
        """
        print("--- HYPOTHESIS TESTING: Structural Missingness ---")

        # 1. Identify Non-Bookings
        non_bookings = df[df['booked'] == 0]
        count_nb = len(non_bookings)
        print(f"Total Non-Bookings (booked=0): {count_nb}")

        # 2. Columns that should be empty for non-bookings
        # We explicitly list the columns found in the audit (95.1% missing)
        suspect_cols = ['price_paid', 'trip_date', 'seats_left', 'load_factor', 'advance_booking_days']

        all_passed = True

        for col in suspect_cols:
            if col not in df.columns:
                continue

            # Count how many are NaN specifically in the non-booking rows
            missing_count = non_bookings[col].isna().sum()

            # The Match: Missing Count must equal Total Non-Bookings
            if missing_count == count_nb:
                print(f"{col}: Perfectly aligned (100% NaN for non-bookings)")
            else:
                diff = count_nb - missing_count
                print(f"{col}: FAILED. {diff} rows have data despite no booking (Data Leak/Error?)")
                all_passed = False

        if all_passed:
            print("CONCLUSION: Missing data is structural (not an error).")
        else:
            print("CONCLUSION: Data corruption detected in booking logs.")

    def fix_structural_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fills missing values that exist purely because a booking didn't happen.
        """
        print("--- ACTION: Fixing Structural Missing Data ---")
        financial_cols = ['price_paid']

        for col in financial_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0)

        print(f"Filled non-booking nulls with 0 for: {financial_cols}")
        print("Skipped 'seats_left' & 'load_factor' to avoid introducing false data.")

        return df

    @staticmethod
    def report_missing_values(df: pd.DataFrame) -> None:
        """
        Calculates and prints the percentage of missing values for every column.
        Useful for verifying that imputation strategies worked.
        """
        # Calculate percentage
        missing_percentages = df.isna().mean() * 100

        # Filter for cleaner output (optional: only show columns with missing data)
        # missing_percentages = missing_percentages[missing_percentages > 0]

        print(f"--- Missing Value Report (Total Rows: {len(df)}) ---")
        print(missing_percentages.sort_values(ascending=False).to_string(float_format="%.2f%%"))

    def impute_safe_prices(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Imputes 'price_shown' for users where no price experiment was active.
        Logic: If discount_flag=0 AND markup_flag=0, Price MUST equal Baseline.
        """
        print("--- ACTION: Imputing Safe Prices (No-Change Group) ---")

        # 1. Define the Safe Mask
        mask_safe = (df['price_shown'].isna()) & (df['discount_flag'] == 0) & (df['markup_flag'] == 0)

        # 2. Apply Imputation
        count = mask_safe.sum()
        if count > 0:
            df.loc[mask_safe, 'price_shown'] = df.loc[mask_safe, 'baseline_price']
            print(f"Imputed {count} values using Baseline Price (Safe Group).")
        else:
            print("No missing values found in the Safe Group.")

        # 3. Status Update
        remaining = df['price_shown'].isna().sum()
        print(f"Remaining Missing Prices: {remaining}")

        return df

    def impute_dynamic_prices(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies a waterfall imputation strategy for 'price_shown'.
        Priority 1: Same Day Mean (High Precision).
        Priority 2: Monthly Mean (High Coverage).
        """
        print("--- ACTION: Imputing Unsafe Prices (Waterfall Strategy) ---")


        # 2. Define Grouping Levels
        # Level 1: Specific (Route + Experiment + Exact Day)
        specific_cols = ['trip_id', 'origin', 'destination', 'transport_type',
                         'duration_min', 'distance_km', 'discount_flag', 'markup_flag', 'visit_date']

        # Level 2: General (Route + Experiment)
        general_cols = ['trip_id', 'origin', 'destination', 'transport_type',
                        'duration_min', 'distance_km', 'discount_flag', 'markup_flag']

        # 3. Pass 1: Day-Specific Imputation
        # We check x.count() > 0 to avoid "Mean of empty slice" warnings
        df['price_shown'] = df.groupby(specific_cols)['price_shown'].transform(
            lambda x: x.fillna(x.mean()) if x.count() > 0 else x
        )
        print("Pass 1 (Daily Specific) applied.")

        print(f"Remaining Missing Prices: {df['price_shown'].isna().sum()}")

        # 4. Pass 2: Monthly General Imputation (Fallback)
        # Catches the 'Singleton' days that Pass 1 missed
        df['price_shown'] = df.groupby(general_cols)['price_shown'].transform(
            lambda x: x.fillna(x.mean())
        )
        print("Pass 2 (Monthly General) applied.")

        # 5. Validation
        print(f"Remaining Missing Prices: {df['price_shown'].isna().sum()}")

        return df

    def diagnose_competitor_price_stability(self, df: pd.DataFrame):
        """
        Tests the hypothesis that prices are more stable within a single day
        than across the entire month.
        """
        print("--- HYPOTHESIS TESTING: Price Stability (Day vs Month) ---")

        # 1. Create a deep copy so we don't mess up the original DF
        work_df = df.copy()

        # 3. Define the Route
        route_cols = ['trip_id', 'origin', 'destination', 'transport_type',
                      'duration_min', 'distance_km', 'discount_flag', 'markup_flag']

        # 4. Calculate Variance using the COPY
        # Scenario A: Month-long variance
        month_std = work_df.groupby(route_cols)['price_shown'].std().mean()

        # Scenario B: Same-day variance
        day_std = work_df.groupby(route_cols + ['visit_date'])['price_shown'].std().mean()

        # 5. The Verdict
        print(f"Average Price Swing (Whole Month): +/- ${month_std:.2f}")
        print(f"Average Price Swing (Same Day):    +/- ${day_std:.2f}")

        if day_std < month_std:
            print("CONFIRMED: Prices are significantly more stable within the same day.")
            print("   Action: We will use 'Visit Day' as a grouping key for imputation.")
        else:
            print("REJECTED: Daily grouping adds no value.")

        # When this function ends, work_df (and the extra column) is destroyed.

    def impute_competitor_prices(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Imputes missing competitor prices using a Route-to-Price lookup map.
        Requires 'competitor_price' to be static per route (verified by diagnosis).
        """
        print("--- ACTION: Imputing Competitor Prices (Lookup Strategy) ---")

        # 1. Build the Lookup Map from valid data
        # We filter for > 0 because 0 represents missing data in this dataset
        valid_comp = df[df['competitor_price'] > 0]
        comp_price_map = valid_comp.groupby('trip_id')['competitor_price'].first().to_dict()

        # 2. Standardize Missing Values
        # Convert 0 to NaN so fillna() works on everything at once
        df['competitor_price'] = df['competitor_price'].replace(0, float('nan'))

        # 3. Apply the Map
        initial_missing = df['competitor_price'].isna().sum()
        df['competitor_price'] = df['competitor_price'].fillna(df['trip_id'].map(comp_price_map))
        final_missing = df['competitor_price'].isna().sum()

        print(f"Imputed {initial_missing - final_missing} values.")
        print(f"Remaining Missing Competitor Prices: {final_missing}")

        return df

    @staticmethod
    def clean_clicked_trip_column(df):
        """
        Cleans the clicked_trip column by logically backfilling
        conversions and imputing remaining NaNs.
        """
        # 1. Identify logical inconsistencies
        mask_converted = (df['added_to_cart'] == 1) | (df['booked'] == 1)

        impossible_rows = df[mask_converted & (df['clicked_trip'] == 0)]
        recoverable_nans = df[mask_converted & (df['clicked_trip'].isna())]

        print(f"CRITICAL ERRORS (Bought but Click=0): {len(impossible_rows)}")
        print(f"RECOVERABLE DATA (Bought but Click=NaN): {len(recoverable_nans)}")

        # 2. Apply Logical Backfill
        df.loc[mask_converted, 'clicked_trip'] = 1

        # 3. Handle remaining NaNs
        remaining_nans = df['clicked_trip'].isna().sum()
        percent_missing = (remaining_nans / len(df)) * 100

        print(f"Remaining NaNs filled with 0: {remaining_nans} ({percent_missing:.2f}%)")
        df['clicked_trip'] = df['clicked_trip'].fillna(0)

        return df

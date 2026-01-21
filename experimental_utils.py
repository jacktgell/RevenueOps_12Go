
import contextlib
import os

from features import FeatureEngineering
from modeling import get_configured_logistic_model, FunnelModeler, ModelArena, XGBoostModel, RandomForestModel, \
    CatBoostModel, TabNetModel, BalancedBaggingModel
from sklearn.metrics import brier_score_loss
from IPython.display import display, Markdown

def run_simple_simulation(models_board, selector, X_test, margin=0.20):
    results = []

    # Define features
    leakage_cols = ['clicked_trip', 'added_to_cart', 'booked']
    technical_drops = leakage_cols + [c for c in X_test.columns if 'trip_id_' in c]
    standard_features = [c for c in X_test.columns if c not in technical_drops]

    cost = 1.0 - margin

    for _, row in models_board.iterrows():
        m_name = row['Model']
        wrapper = row['Object']
        actual_model = wrapper.model if hasattr(wrapper, 'model') else wrapper

        feats = selector.selection_results['booked']['features'] if (
                    "Stepwise" in m_name and 'booked' in selector.selection_results) else standard_features

        # 1. BASELINE
        X_base = X_test[feats].copy()
        base_input = X_base.values if "TabNet" in m_name else X_base
        probs_base = actual_model.predict_proba(base_input)[:, 1]

        base_vol = probs_base.sum()
        base_profit = (probs_base * (1.0 - cost)).sum()

        # 2. RUN SCENARIOS
        for scenario_name, multiplier in [("Price Up (+10%)", 1.10), ("Price Down (-10%)", 0.90)]:
            X_sim = X_base.copy()
            if 'price_shown' in X_sim.columns:
                X_sim['price_shown'] *= multiplier

            sim_input = X_sim.values if "TabNet" in m_name else X_sim
            probs_sim = actual_model.predict_proba(sim_input)[:, 1]

            sim_vol = probs_sim.sum()
            sim_profit = (probs_sim * (multiplier - cost)).sum()

            results.append({
                'Model': m_name,
                'Scenario': scenario_name,
                'Volume_Impact_%': ((sim_vol - base_vol) / base_vol) * 100,
                'Profit_Impact_%': ((sim_profit - base_profit) / base_profit) * 100
            })

    return pd.DataFrame(results)


class LeaderboardReporter:
    def __init__(self, targets):
        """
        Args:
            targets (list): List of targets to process (e.g. ['clicked_trip', ...])
        """
        self.targets = targets

    def update_and_display(self, current_boards, selection_results, test_df, train_columns, row_name=None):
        """
        Updates leaderboards with new optimized models and displays them.

        Args:
            current_boards (dict): Dictionary of existing leaderboard DataFrames.
            selection_results (dict): Dictionary containing new models from StepwiseSelector.
            test_df (pd.DataFrame): The test dataset (ground truth).
            train_columns (list/Index): Columns from training data (used for filtering logic).

        Returns:
            dict: The updated leaderboards dictionary.
        """
        # Start with a copy so we don't mutate the original dictionary in place immediately
        updated_boards = current_boards.copy()
        if row_name == None:
            row_name = 'Optimized TabNet (Stepwise)'



        leakage_cols = ['clicked_trip', 'added_to_cart', 'booked']

        # Determine drop columns (ID columns + leakage columns)
        technical_drops = leakage_cols + [c for c in train_columns if 'trip_id_' in c]
        technical_drops = [c for c in technical_drops if c in test_df.columns]

        for target in self.targets:
            # 1. Get or Create Board if it doesn't exist
            board = updated_boards.get(target, pd.DataFrame(columns=['Model', 'PR_AUC', 'Brier_Score', 'Object']))

            # 2. Process New Model (if available in selection_results)
            if target in selection_results:
                res = selection_results[target]
                model = res['model']
                feats = res['features']

                # Prepare Data (Slice only, no training)
                X_test_curr = test_df.drop(columns=technical_drops)
                y_test_curr = test_df[target]

                # Predict
                y_prob = model.predict_proba(X_test_curr[feats])[:, 1]

                # Create Entry
                new_row = {
                    "Model": row_name,
                    "PR_AUC": average_precision_score(y_test_curr, y_prob),
                    "Brier_Score": brier_score_loss(y_test_curr, y_prob),
                    "Object": model
                }

                # Merge: Remove old version of this model (if exists), then append new
                if not board.empty:
                    board = board[board['Model'] != row_name]

                board = pd.concat([board, pd.DataFrame([new_row])], ignore_index=True)

                # 3. Calculate Lift & Sort
                baseline_prob = y_test_curr.mean()
                board['Lift_vs_Random'] = board['PR_AUC'] / baseline_prob
                board = board.sort_values(by="PR_AUC", ascending=False).reset_index(drop=True)

                # Save back to dict
                updated_boards[target] = board

                # 4. Display (Side Effect)
                self._display_table(target, board)
            else:
                if not board.empty:
                    self._display_table(target, board)

        return updated_boards

    def _display_table(self, target, board):
        """Helper to handle the Pandas Styling."""
        display(Markdown(f"### Leaderboard: {target.replace('_', ' ').title()}"))

        # Select columns to show
        cols = ['Model', 'PR_AUC', 'Brier_Score', 'Lift_vs_Random']

        # Check if columns exist before styling (in case board is empty)
        if not board.empty and all(c in board.columns for c in cols):
            styler = (board[cols].style
            .background_gradient(cmap='Greens', subset=['PR_AUC'])
            .highlight_min(subset=['Brier_Score'], color='lightgreen')
            .format({
                'PR_AUC': "{:.4f}",
                'Brier_Score': "{:.4f}",
                'Lift_vs_Random': "{:.2f}x"
            }))
            display(styler)
        else:
            display(board)


class TabNetStepwiseSelector:
    def __init__(self, X_train, y_train, X_test, y_test, initial_features=None):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.available_features = list(X_train.columns)
        self.current_features = initial_features if initial_features else []
        self.best_score = -1
        self.best_model = None
        self.selection_results = {}

    def get_score(self, features):
        if not features: return 0
        wrapper = TabNetModel()
        try:
            with contextlib.redirect_stdout(open(os.devnull, 'w')):
                wrapper.train(self.X_train[features], self.y_train)
                metrics = wrapper.evaluate(self.X_test[features], self.y_test)
            return metrics['PR_AUC']
        except:
            return 0

    def run(self, max_cycles=5, improvement_tol=0.00001):
        print(f"--- Starting TabNet Selection (Inclusive Strategy) ---")

        if self.current_features:
            self.best_score = self.get_score(self.current_features)
            print(f"Baseline Score: {self.best_score:.5f}")

        for cycle in range(1, max_cycles + 1):
            print(f"\n=== CYCLE {cycle} ===")
            changed = False

            # --- ADD PHASE (Inclusive) ---
            # We iterate through candidates. If one works, we keep it immediately.
            candidates = [f for f in self.available_features if f not in self.current_features]
            n_candidates = len(candidates)

            for i, f in enumerate(candidates):
                # Test feature against current set
                score = self.get_score(self.current_features + [f])

                if score > self.best_score + improvement_tol:
                    self.current_features.append(f)
                    self.best_score = score
                    changed = True
                    print(f"[ADDED] Col {f} | Feature {i + 1}/{n_candidates} | Cycle {cycle} | Score: {score:.5f}")
                else:
                    # Optional: Use end='\r' to keep skips on one line to reduce clutter
                    print(
                        f"[SKIP]  Col {f} | Feature {i + 1}/{n_candidates} | Cycle {cycle} | Score: {score:.5f} (Best: {self.best_score:.5f})")

            print("")

            features_to_check = self.current_features[:]
            n_features = len(features_to_check)

            for i, f in enumerate(features_to_check):
                # Test removing feature
                score = self.get_score([x for x in self.current_features if x != f])

                if score > self.best_score:
                    self.current_features.remove(f)
                    self.best_score = score
                    changed = True
                    print(f"[DROP]  Col {f} | Feature {i + 1}/{n_features} | Cycle {cycle} | Score: {score:.5f}")
                else:
                    print(
                        f"[KEEP]  Col {f} | Feature {i + 1}/{n_features} | Cycle {cycle} | Score: {score:.5f} (Best: {self.best_score:.5f})")

            print("")

            if not changed:
                print("\nConvergence reached (No additions or drops improved the model).")
                break

        # --- FINAL STEP ---
        print(f"\nTraining final model on {len(self.current_features)} features...")
        self.best_model = TabNetModel()
        with contextlib.redirect_stdout(open(os.devnull, 'w')):
            self.best_model.train(self.X_train[self.current_features], self.y_train)

        target_name = self.y_train.name if hasattr(self.y_train, 'name') else 'booked'
        self.selection_results = {
            target_name: {
                'model': self.best_model,
                'features': self.current_features,
                'name': 'TabNet (Stepwise Inclusive)'
            }
        }

        return self.best_model

from catboost import CatBoostClassifier
import pandas as pd
from sklearn.metrics import average_precision_score
import copy


class CatBoostStepwiseSelector:
    """
    Performs Stepwise Feature Selection (Bidirectional) using CatBoost.
    Matches the logic of TabNetStepwiseSelector:
    1. Inclusive Add Phase: Iterates candidates; if one improves score, it's kept immediately.
    2. Drop Phase: Iterates current features; if dropping one improves score, it's removed immediately.
    """

    def __init__(self, X_train, y_train, X_test, y_test, initial_features=None):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

        # Pool of all available columns from the training set
        self.available_features = list(X_train.columns)

        # Start with initial features or empty list
        self.current_features = initial_features if initial_features else []

        self.best_score = -1
        self.best_model = None
        self.selection_results = {}

    def get_score(self, features):
        """
        Trains a quick CatBoost model on specific features and returns PR-AUC.
        """
        if not features: return 0

        # Fast config for selection speed
        model = CatBoostClassifier(
            iterations=500,
            learning_rate=0.03,  # Slightly higher for faster convergence during selection
            depth=6,
            loss_function='Logloss',
            auto_class_weights='Balanced',
            verbose=False,
            early_stopping_rounds=20,
            random_state=42,
            allow_writing_files=False
        )

        try:
            # Subset data
            X_tr_sub = self.X_train[features]
            X_te_sub = self.X_test[features]

            # Train with Early Stopping on Test set (standard for boosting selection)
            model.fit(
                X_tr_sub, self.y_train,
                eval_set=(X_te_sub, self.y_test),
                verbose=False
            )

            # Evaluate
            y_prob = model.predict_proba(X_te_sub)[:, 1]
            return average_precision_score(self.y_test, y_prob)

        except Exception as e:
            print(f"Error in get_score: {e}")
            return 0

    def run(self, max_cycles=5, improvement_tol=0.00001):
        print(f"--- Starting CatBoost Selection (Inclusive Strategy) ---")

        # 1. Establish Baseline
        if self.current_features:
            self.best_score = self.get_score(self.current_features)
            print(f"Baseline Score: {self.best_score:.5f}")
        else:
            self.best_score = 0

        # 2. Main Cycle Loop
        for cycle in range(1, max_cycles + 1):
            print(f"\n=== CYCLE {cycle} ===")
            changed = False

            # --- ADD PHASE (Inclusive) ---
            # Identify candidates not yet in the model
            candidates = [f for f in self.available_features if f not in self.current_features]
            n_candidates = len(candidates)

            for i, f in enumerate(candidates):
                # Test feature against current set
                score = self.get_score(self.current_features + [f])

                if score > self.best_score + improvement_tol:
                    self.current_features.append(f)
                    self.best_score = score
                    changed = True
                    print(f"[ADDED] Col {f} | Feature {i + 1}/{n_candidates} | Cycle {cycle} | Score: {score:.5f}")
                else:
                    # Optional: Print skips (comment out to reduce noise)
                    # print(f"[SKIP]  Col {f} | Feature {i + 1}/{n_candidates} | Cycle {cycle} | Score: {score:.5f} (Best: {self.best_score:.5f})")
                    pass

            print(f"   (End of Add Phase. Current Features: {len(self.current_features)})")

            # --- DROP PHASE (Backward Elimination) ---
            # Check if we can remove any feature without hurting performance
            features_to_check = self.current_features[:]
            n_features = len(features_to_check)

            for i, f in enumerate(features_to_check):
                # Test removing feature
                remaining_features = [x for x in self.current_features if x != f]
                score = self.get_score(remaining_features)

                # If score improves (or stays same) after dropping, we drop it.
                # Note: 'score > self.best_score' ensures strict improvement.
                # Use 'score >= self.best_score' if you prefer simpler models when performance is equal.
                if score > self.best_score:
                    self.current_features.remove(f)
                    self.best_score = score
                    changed = True
                    print(f"[DROP]  Col {f} | Feature {i + 1}/{n_features} | Cycle {cycle} | Score: {score:.5f}")
                else:
                    pass
                    # print(f"[KEEP]  Col {f} | Feature {i + 1}/{n_features} | Cycle {cycle} | Score: {score:.5f}")

            if not changed:
                print("\nConvergence reached (No additions or drops improved the model).")
                break

        # --- FINAL STEP ---
        print(f"\nTraining final model on {len(self.current_features)} features...")

        # Train one final robust model
        self.best_model = CatBoostClassifier(
            iterations=1000,  # Full iterations for final model
            learning_rate=0.01,
            depth=6,
            loss_function='Logloss',
            auto_class_weights='Balanced',
            verbose=False,
            early_stopping_rounds=50,
            random_state=42,
            allow_writing_files=False
        )

        self.best_model.fit(
            self.X_train[self.current_features], self.y_train,
            eval_set=(self.X_test[self.current_features], self.y_test),
            verbose=False
        )

        target_name = self.y_train.name if hasattr(self.y_train, 'name') else 'booked'
        self.selection_results = {
            target_name: {
                'model': self.best_model,
                'features': self.current_features,
                'name': 'CatBoost (Stepwise Inclusive)'
            }
        }

        return self.best_model


class TransportExperimentRunner:
    """
    Orchestrates the end-to-end experiment for a single transport mode:
    1. Data Preparation (Cleaning, Feature Eng, Splitting)
    2. Linear Diagnostics
    3. Model Competition (Standard)
    4. Stepwise Selection (Optimized)
    5. Reporting & Simulation
    """

    def __init__(self, mode, raw_df, config):
        self.mode = mode
        self.raw_df = raw_df
        self.config = config
        self.train_df = None
        self.test_df = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.selector = None
        self.leaderboard = None

    def prepare_data(self):
        """Encapsulates all data wrangling logic."""
        print(f"   > Preparing data for {self.mode}...")

        # 1. Slice
        mode_df = self.raw_df[self.raw_df['transport_type'] == self.mode].copy()

        # 2. Update Configs (Remove transport_type)
        curr_encode = [c for c in self.config['cols_to_encode'] if c != 'transport_type']
        curr_drop = self.config['cols_to_drop'] + ['transport_type']

        # 3. Feature Engineering
        modeling_df = FeatureEngineering.prepare_modeling_data(mode_df, curr_encode, curr_drop)

        # Computeds
        modeling_df['price_per_km'] = modeling_df['price_shown'] / (modeling_df['distance_km'] + 1)
        modeling_df['valuable_deal'] = modeling_df['price_ratio_vs_competitor'] * modeling_df['popularity_score']

        # Cleanup
        if 'transport_type' in modeling_df.columns:
            modeling_df.drop(columns=['transport_type'], inplace=True)

        # 4. Split & Scale
        self.train_df, self.test_df = FeatureEngineering.global_random_split(modeling_df)
        self.train_df, self.test_df = FeatureEngineering.scale_numeric_features(
            self.train_df, self.test_df, self.config['cols_to_scale']
        )

        # 5. Define Matrices (Booked Target)
        target = 'booked'
        leakage = ['clicked_trip', 'added_to_cart', 'booked']
        tech_drops = leakage + [c for c in self.train_df.columns if 'trip_id_' in c]

        self.X_train = self.train_df.drop(columns=tech_drops)
        self.y_train = self.train_df[target]
        self.X_test = self.test_df.drop(columns=tech_drops)
        self.y_test = self.test_df[target]

    def run_linear_diagnostics(self):
        """Runs the FunnelModeler."""
        print(f"   > Running Linear Diagnostics...")
        funnel = FunnelModeler()
        funnel.train_all_stages(self.train_df, self.test_df)

    def run_standard_competition(self):
        """Runs the standard 'Horse Race'."""
        print(f"   > Running Standard Model Competition...")
        # Baseline
        base_model = get_configured_logistic_model()
        base_model.fit(self.X_train, self.y_train)
        y_prob = base_model.predict_proba(self.X_test)[:, 1]

        base_metrics = {
            'pr_auc': average_precision_score(self.y_test, y_prob),
            'brier': brier_score_loss(self.y_test, y_prob)
        }

        competitors = [
            ("XGBoost", XGBoostModel()),
            ("Random Forest", RandomForestModel()),
            ("CatBoost", CatBoostModel()),
            ("TabNet", TabNetModel()),
            ("Balanced Bagging", BalancedBaggingModel())
        ]

        self.leaderboard = ModelArena.run_competition(
            competitors, self.X_train, self.y_train, self.X_test, self.y_test,
            base_metrics, base_model
        )

        # Display Standard Board
        display(Markdown(f"#### Standard Leaderboard ({self.mode})"))
        base_pr = self.y_test.mean()
        self.leaderboard['Lift_vs_Random'] = self.leaderboard['PR_AUC'] / base_pr
        display(self.leaderboard[['Model', 'PR_AUC', 'Lift_vs_Random']].head(3))

    def run_stepwise_optimization(self):
        """Runs the CatBoost Stepwise Selector."""
        print(f"   > Running Stepwise Optimization (CatBoost)...")
        initial_feats = list(self.X_train.columns)[0:1]

        self.selector = CatBoostStepwiseSelector(
            self.X_train, self.y_train, self.X_test, self.y_test, initial_feats
        )
        self.selector.run(max_cycles=3)

    def generate_final_report(self):
        """Updates leaderboard and runs simulation."""
        print(f"   > Generating Final Report...")

        # 1. Update Leaderboard
        # Re-create baseline board for reporter context
        baseline = get_configured_logistic_model()
        baseline.fit(self.X_train, self.y_train)
        y_prob = baseline.predict_proba(self.X_test)[:, 1]

        base_board = pd.DataFrame([{
            'Model': 'Linear (Baseline)',
            'PR_AUC': average_precision_score(self.y_test, y_prob),
            'Brier_Score': brier_score_loss(self.y_test, y_prob),
            'Object': baseline
        }])

        reporter = LeaderboardReporter(targets=['booked'])
        final_boards = reporter.update_and_display(
            current_boards={'booked': base_board},
            selection_results=self.selector.selection_results,
            test_df=self.test_df,
            train_columns=self.train_df.columns,
            row_name='CatBoost (Stepwise)'
        )

        # 2. Run Simulation
        sim_df = run_simple_simulation(
            models_board=pd.DataFrame([{'Model': 'CatBoost (Stepwise)', 'Object': self.selector.best_model}]),
            selector=self.selector,
            X_test=self.test_df
        )

        # Calc Elasticity
        elasticity = []
        for _, row in sim_df.iterrows():
            if "Up" in row['Scenario']:
                e = row['Volume_Impact_%'] / 10
            elif "Down" in row['Scenario']:
                e = row['Volume_Impact_%'] / -10
            else:
                e = 0
            elasticity.append(e)
        sim_df['Elasticity'] = elasticity

        display(Markdown(f"#### Elasticity Simulation ({self.mode})"))
        display(sim_df.style.background_gradient(cmap='RdYlGn', subset=['Profit_Impact_%']).format("{:.2f}"))




def prepare_transport_data(raw_df, mode, config):
    """
    Handles slicing, cleaning, encoding, and splitting for a specific transport mode.
    """
    # 1. Slice
    mode_df = raw_df[raw_df['transport_type'] == mode].copy()

    # 2. Config Configs (Remove 'transport_type')
    curr_encode = [c for c in config['cols_to_encode'] if c != 'transport_type']
    curr_drop = config['cols_to_drop'] + ['transport_type']

    # 3. Feature Engineering
    modeling_df = FeatureEngineering.prepare_modeling_data(mode_df, curr_encode, curr_drop)

    # 4. Computed Features
    modeling_df['price_per_km'] = modeling_df['price_shown'] / (modeling_df['distance_km'] + 1)
    modeling_df['valuable_deal'] = modeling_df['price_ratio_vs_competitor'] * modeling_df['popularity_score']

    # 5. Cleanup
    if 'transport_type' in modeling_df.columns:
        modeling_df.drop(columns=['transport_type'], inplace=True)

    # 6. Split & Scale
    train_df, test_df = FeatureEngineering.global_random_split(modeling_df)
    train_df, test_df = FeatureEngineering.scale_numeric_features(
        train_df, test_df, config['cols_to_scale']
    )

    return train_df, test_df
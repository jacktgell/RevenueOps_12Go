
import contextlib
import os
from modeling import TabNetModel
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss
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

    def update_and_display(self, current_boards, selection_results, test_df, train_columns):
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
                    "Model": "Optimized TabNet (Stepwise)",
                    "PR_AUC": average_precision_score(y_test_curr, y_prob),
                    "Brier_Score": brier_score_loss(y_test_curr, y_prob),
                    "Object": model
                }

                # Merge: Remove old version of this model (if exists), then append new
                if not board.empty:
                    board = board[board['Model'] != "Optimized TabNet (Stepwise)"]

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
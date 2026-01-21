import contextlib
import os
import warnings

from tqdm import TqdmWarning
warnings.filterwarnings("ignore", module="pytorch_tabnet.callbacks")
import torch
from pytorch_tabnet.tab_model import TabNetClassifier
from pytorch_tabnet.metrics import Metric
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
import xgboost as xgb
from catboost import CatBoostClassifier
from imblearn.ensemble import BalancedBaggingClassifier
warnings.filterwarnings("ignore", module="pytorch_tabnet.callbacks")

def get_configured_logistic_model():
    """
    Factory function for the Linear Baseline.

    SOLVER: liblinear (Stable + L1 support)
    NOTE: n_jobs must be None because liblinear is single-threaded.
    """
    params = {
        'penalty': 'l1',
        'solver': 'liblinear',
        'C': 0.1,
        'class_weight': 'balanced',
        'random_state': 42,
        'max_iter': 10000,
        'n_jobs': None
    }

    return LogisticRegression(**params)


# ------------------------------------------------

class FunnelModeler:
    def __init__(self):
        self.models = {}
        self.results = {}

    def train_all_stages(self, train_df, test_df):
        targets = ['clicked_trip', 'added_to_cart', 'booked']
        feature_cols = [c for c in train_df.columns if c not in targets]

        for target in targets:
            X_train = train_df[feature_cols]
            y_train = train_df[target]
            X_test = test_df[feature_cols]
            y_test = test_df[target]

            model = get_configured_logistic_model()

            model.fit(X_train, y_train)

            y_prob = model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_prob)
            pr_auc = average_precision_score(y_test, y_prob)
            brier = brier_score_loss(y_test, y_prob)

            self.models[target] = model
            self.results[target] = {
                'auc': auc,
                'pr_auc': pr_auc,
                'brier': brier,
                'coefs': dict(zip(feature_cols, model.coef_[0]))
            }


class XGBoostModel:
    def __init__(self):
        self.model = None

    def train(self, X_train, y_train):
        feature_names = X_train.columns.tolist()
        constraints = [0] * len(feature_names)
        for i, col in enumerate(feature_names):
            if 'price_ratio' in col:
                constraints[i] = -1

        self.model = xgb.XGBClassifier(
            objective='binary:logistic',
            n_estimators=100, max_depth=3, learning_rate=0.05,
            scale_pos_weight=20,
            monotone_constraints=tuple(constraints),
            n_jobs=-1, random_state=42
        )
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        y_prob = self.model.predict_proba(X_test)[:, 1]
        metrics = {
            'ROC_AUC': roc_auc_score(y_test, y_prob),
            'PR_AUC': average_precision_score(y_test, y_prob),
            'Brier_Score': brier_score_loss(y_test, y_prob)
        }
        return metrics


class RandomForestModel:
    def __init__(self):
        self.model = None

    def train(self, X_train, y_train):
        self.model = RandomForestClassifier(
            n_estimators=200, max_depth=6, min_samples_leaf=10,
            class_weight='balanced', n_jobs=-1, random_state=42
        )
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        y_prob = self.model.predict_proba(X_test)[:, 1]
        metrics = {
            'ROC_AUC': roc_auc_score(y_test, y_prob),
            'PR_AUC': average_precision_score(y_test, y_prob),
            'Brier_Score': brier_score_loss(y_test, y_prob)
        }
        return metrics


class CatBoostModel:
    def __init__(self):
        self.model = None

    def train(self, X_train, y_train):
        if CatBoostClassifier is None: return

        feature_names = X_train.columns.tolist()
        monotone_constraints = {}
        for col in feature_names:
            if 'price_ratio' in col:
                monotone_constraints[col] = -1

        self.model = CatBoostClassifier(
            iterations=1000,
            learning_rate=0.01,
            depth=6,
            loss_function='Logloss',
            auto_class_weights='Balanced',
            monotone_constraints=monotone_constraints,
            verbose=False,
            early_stopping_rounds=50,
            random_state=42
        )
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        if self.model is None: return {'PR_AUC': 0, 'Brier_Score': 1}
        y_prob = self.model.predict_proba(X_test)[:, 1]
        metrics = {
            'ROC_AUC': roc_auc_score(y_test, y_prob),
            'PR_AUC': average_precision_score(y_test, y_prob),
            'Brier_Score': brier_score_loss(y_test, y_prob)
        }
        return metrics


class BalancedBaggingModel:
    def __init__(self):
        self.model = None

    def train(self, X_train, y_train):
        if BalancedBaggingClassifier is None: return

        self.model = BalancedBaggingClassifier(
            estimator=DecisionTreeClassifier(max_depth=6),
            n_estimators=50,
            sampling_strategy='auto',
            random_state=42,
            n_jobs=-1
        )
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        if self.model is None: return {'PR_AUC': 0, 'Brier_Score': 1}
        y_prob = self.model.predict_proba(X_test)[:, 1]
        metrics = {
            'ROC_AUC': roc_auc_score(y_test, y_prob),
            'PR_AUC': average_precision_score(y_test, y_prob),
            'Brier_Score': brier_score_loss(y_test, y_prob)
        }
        return metrics


class SimulationEngine:
    def __init__(self, model_obj):
        self.model = model_obj.model if hasattr(model_obj, 'model') else model_obj

    def simulate_scenarios(self, df):
        scenarios = {'baseline': 1.0, 'price_up': 1.10, 'price_down': 0.90}
        results = []

        for name, multiplier in scenarios.items():
            sim_df = df.copy()
            if 'price_ratio_vs_competitor' in sim_df.columns:
                sim_df['price_ratio_vs_competitor'] *= multiplier

            prob = self.model.predict_proba(sim_df)[:, 1]
            avg_conv = prob.mean()

            results.append({
                'Scenario': name,
                'Multiplier': multiplier,
                'Conversion_Rate': avg_conv,
                'Revenue_Index': multiplier * avg_conv * 1000
            })
        return pd.DataFrame(results)

    def recommend_policy(self, sim_results):
        base_rev = sim_results.loc[sim_results['Scenario'] == 'baseline', 'Revenue_Index'].values[0]
        sim_results['Impact_%'] = (sim_results['Revenue_Index'] - base_rev) / base_rev * 100
        print("--- Simulation Outcome ---")
        print(sim_results.round(4).to_string(index=False))
        best = sim_results.loc[sim_results['Revenue_Index'].idxmax()]
        print(f"RECOMMENDATION: {best['Scenario'].upper()} (Impact: {best['Impact_%']:.2f}%)")


class ModelArena:
    @staticmethod
    def run_competition(competitors, X_train, y_train, X_test, y_test, linear_metrics, linear_model_obj):
        results = []
        results.append({
            "Model": "Linear (Baseline)",
            "PR_AUC": linear_metrics['pr_auc'],
            "Brier_Score": linear_metrics['brier'],
            "Object": linear_model_obj
        })

        for name, model_obj in competitors:
            if name == "Linear (Baseline)": continue
            try:
                model_obj.train(X_train, y_train)
                metrics = model_obj.evaluate(X_test, y_test)
                results.append({
                    "Model": name,
                    "PR_AUC": metrics['PR_AUC'],
                    "Brier_Score": metrics['Brier_Score'],
                    "Object": model_obj
                })
            except Exception as e:
                print(f"(Error: {str(e)[:50]}...)")

        return pd.DataFrame(results).sort_values(by="PR_AUC", ascending=False)


class PRAUC(Metric):
    def __init__(self):
        self._name = "pr_auc"
        self._maximize = True

    def __call__(self, y_true, y_score):
        return average_precision_score(y_true, y_score[:, 1])


class TabNetModel:
    def __init__(self):
        self.model = None

    def train(self, X_train, y_train):
        # 1. Prepare Data
        X_vals = X_train.values
        y_vals = y_train.values

        # Split internal validation set (20%)
        # This is a validation set the test set is outside the scope of this instance
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_vals, y_vals, test_size=0.2, random_state=42, stratify=y_vals
        )

        # 2. Define Model
        self.model = TabNetClassifier(
            optimizer_fn=torch.optim.Adam,
            optimizer_params=dict(lr=2e-2),
            scheduler_params={"step_size": 10, "gamma": 0.9},
            scheduler_fn=torch.optim.lr_scheduler.StepLR,
            mask_type='entmax',
            lambda_sparse=5e-2,
            verbose=0,
            seed=42
        )

        # 3. Fit (Silenced)
        with contextlib.redirect_stdout(open(os.devnull, 'w')):
            self.model.fit(
                X_train=X_tr, y_train=y_tr,
                eval_set=[(X_val, y_val)],
                eval_name=['val'],
                eval_metric=[PRAUC],
                weights=1,
                max_epochs=150,
                patience=30,
                batch_size=1024,
                virtual_batch_size=128,
                num_workers=0,
                drop_last=False
            )

    def evaluate(self, X_test, y_test):
        if self.model is None: return {'PR_AUC': 0, 'Brier_Score': 1}

        # Helper to get probs safely
        y_prob = self.predict_proba(X_test)[:, 1]

        return {
            'PR_AUC': average_precision_score(y_test, y_prob),
            'Brier_Score': brier_score_loss(y_test, y_prob)
        }

    def predict_proba(self, X):
        """Passes through to internal model, handling DataFrame conversion."""
        if hasattr(X, 'values'):
            X = X.values
        return self.model.predict_proba(X)



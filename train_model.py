"""
Model training module for Fake News Detection.

This module handles training, evaluation, and comparison of Logistic Regression
and Passive Aggressive Classifier models.
"""

import pickle
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)

from data_prep import SyntheticDataGenerator, prepare_data_for_training, split_data
from utils import create_vectorizer, TrainingFailedError


class ModelTrainer:
    def __init__(self):
        self.generator = SyntheticDataGenerator()
        self.vectorizer = None
        self.models = {}
        self.scores = {}
        self.best_model_name = None
        self.best_model = None

    def train_models(self):
        print("\n" + "=" * 70)
        print("FAKE NEWS DETECTION - MODEL TRAINING")
        print("=" * 70 + "\n")

        # Step 1: Generate synthetic data
        print("[1/6] Generating synthetic dataset...")
        texts, labels = self.generator.generate_dataset(samples_per_class=1000)
        print(f"    ✓ Generated {len(texts)} samples")
        print(f"    ✓ Real samples: {sum(1 for l in labels if l == 0)}")
        print(f"    ✓ Fake samples: {sum(1 for l in labels if l == 1)}\n")

        # Step 2: Preprocess texts
        print("[2/6] Preprocessing texts...")
        texts = prepare_data_for_training(texts, labels)
        print("    ✓ Texts preprocessed\n")

        # Step 3: Split first
        print("[3/6] Splitting data (80/20)...")
        X_train, X_test, y_train, y_test = split_data(texts, labels, train_size=0.8)
        print(f"    ✓ Training samples: {len(X_train)}")
        print(f"    ✓ Testing samples: {len(X_test)}\n")

        # Step 4: Vectorize
        print("[4/6] Vectorizing with TF-IDF...")
        self.vectorizer = create_vectorizer(max_features=5000, ngram_range=(1, 2))

        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)

        print("    ✓ Vectorization complete")
        print(f"    ✓ Features extracted: {X_train_vec.shape[1]}\n")

        # Step 5: Train models
        print("[5/6] Training models...")
        self._train_logistic_regression(X_train_vec, y_train)
        self._train_passive_aggressive(X_train_vec, y_train)

        # Step 6: Evaluate
        print("\n[6/6] Evaluating models...")
        self._evaluate_models(X_train_vec, X_test_vec, y_train, y_test)

        # Select best
        self._select_best_model()

        # Save
        self._save_models()

        print("\n" + "=" * 70)
        print("TRAINING COMPLETE")
        print("=" * 70 + "\n")

    def _train_logistic_regression(self, X_train, y_train):
        print("    Training Logistic Regression...")
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_train, y_train)
        self.models['Logistic Regression'] = model
        print("    ✓ Logistic Regression trained")

    def _train_passive_aggressive(self, X_train, y_train):
        print("    Training Passive Aggressive Classifier...")
        model = PassiveAggressiveClassifier(max_iter=1000, random_state=42)
        model.fit(X_train, y_train)
        self.models['Passive Aggressive'] = model
        print("    ✓ Passive Aggressive Classifier trained")

    def _evaluate_models(self, X_train, X_test, y_train, y_test):
        for model_name, model in self.models.items():
            print(f"\n    {model_name}:")

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_accuracy = accuracy_score(y_train, y_train_pred)
            test_accuracy = accuracy_score(y_test, y_test_pred)
            test_precision = precision_score(y_test, y_test_pred)
            test_recall = recall_score(y_test, y_test_pred)
            test_f1 = f1_score(y_test, y_test_pred)

            # Safe ROC-AUC
            if hasattr(model, "predict_proba"):
                y_scores = model.predict_proba(X_test)[:, 1]
            elif hasattr(model, "decision_function"):
                y_scores = model.decision_function(X_test)
            else:
                y_scores = y_test_pred

            test_auc = roc_auc_score(y_test, y_scores)

            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')

            self.scores[model_name] = {
                'train_accuracy': train_accuracy,
                'test_accuracy': test_accuracy,
                'precision': test_precision,
                'recall': test_recall,
                'f1': test_f1,
                'auc': test_auc,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }

            print(f"      Train Accuracy: {train_accuracy:.4f}")
            print(f"      Test Accuracy:  {test_accuracy:.4f}")
            print(f"      Precision:      {test_precision:.4f}")
            print(f"      Recall:         {test_recall:.4f}")
            print(f"      F1-Score:       {test_f1:.4f}")
            print(f"      ROC-AUC:        {test_auc:.4f}")
            print(f"      CV (Mean±Std):  {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

            cm = confusion_matrix(y_test, y_test_pred)
            print("\n      Confusion Matrix:")
            print(f"      [{cm[0,0]:4d}  {cm[0,1]:4d}]")
            print(f"      [{cm[1,0]:4d}  {cm[1,1]:4d}]")

            print("\n      Classification Report:")
            report = classification_report(
                y_test,
                y_test_pred,
                target_names=['Real', 'Fake'],
                digits=4
            )
            for line in report.split('\n'):
                print(f"      {line}")

    def _select_best_model(self):
        print("\n    Selecting best model...")

        self.best_model_name = max(
            self.scores,
            key=lambda name: self.scores[name]['f1']
        )

        self.best_model = self.models[self.best_model_name]

        print(f"\n    Best Model: {self.best_model_name}")
        print(f"    F1-Score: {self.scores[self.best_model_name]['f1']:.4f}")

    def _save_models(self):
        print(f"\n    Saving {self.best_model_name} model...")

        try:
            with open('vectorizer.pkl', 'wb') as f:
                pickle.dump(self.vectorizer, f)

            with open('model.pkl', 'wb') as f:
                pickle.dump(self.best_model, f)

            with open('model_metadata.pkl', 'wb') as f:
                pickle.dump({
                    'model_name': self.best_model_name,
                    'metrics': self.scores[self.best_model_name]
                }, f)

            print("    ✓ vectorizer.pkl saved")
            print("    ✓ model.pkl saved")
            print("    ✓ model_metadata.pkl saved")

        except Exception as e:
            raise TrainingFailedError(f"Failed to save models: {str(e)}")


def main():
    try:
        trainer = ModelTrainer()
        trainer.train_models()

        print("\n✅ Training completed successfully")
        print("Generated files:")
        print(" - model.pkl")
        print(" - vectorizer.pkl")
        print(" - model_metadata.pkl")

    except Exception as e:
        print(f"\nTraining failed: {e}")
        raise


if __name__ == "__main__":
    main()
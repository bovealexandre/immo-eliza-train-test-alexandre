import optuna
import pickle
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score


def regressor(X_train, X_test, y_train, y_test):
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 10, 200),
            "max_depth": trial.suggest_int("max_depth", 2, 32),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 150),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 60),
        }

        model = RandomForestRegressor(**params, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)

        return r2

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10)

    best_params = study.best_params
    best_model = RandomForestRegressor(
        n_estimators=best_params["n_estimators"],
        max_depth=best_params["max_depth"],
        min_samples_split=best_params["min_samples_split"],
        min_samples_leaf=best_params["min_samples_leaf"],
        random_state=0,
    )
    best_model.fit(X_train, y_train)

    final_predictions = best_model.predict(X_test)
    final_mse = r2_score(y_test, final_predictions)
    print(f"Final r² Score on Test Data: {final_mse:.4f}")

    with open("model.pkl", "wb") as model_file:
        pickle.dump(best_model, model_file)

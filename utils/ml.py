from sklearn.model_selection import train_test_split
import pickle
from sklearn.metrics import r2_score, mean_squared_error
from catboost import CatBoostRegressor

import optuna


def regressor(df):
    X = df.drop(columns=["Price"])
    y = df["Price"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.08, random_state=42
    )

    def train_and_evaluate_catboost(params):
        model = CatBoostRegressor(**params, silent=True)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        r2 = r2_score(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        return model, r2, mse

    def objective(trial):
        params = {
            "iterations": 1000,
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
            "depth": trial.suggest_int("depth", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.05, 1.0),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.05, 1.0),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 100),
        }

        model, r2, mse = train_and_evaluate_catboost(params)

        if r2 > study.user_attrs.get("best_r2", -1):
            with open("best_catboost_model.pkl", "wb") as model_file:
                pickle.dump(model, model_file)
            study.set_user_attr("best_r2", r2)
            study.set_user_attr("best_mse", mse)

        return mse

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=30)

    best_r2 = study.user_attrs.get("best_r2", -1)
    best_mse = study.user_attrs.get("best_mse", -1)
    print("Best R-squared:", best_r2)
    print("Best MSE:", best_mse)

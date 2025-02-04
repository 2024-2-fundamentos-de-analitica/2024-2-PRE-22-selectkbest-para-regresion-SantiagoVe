def load_data():

    import pandas as pd

    dataset = pd.read_csv("../files/input/auto_mpg.csv")
    dataset = dataset.dropna()
    dataset["Origin"] = dataset["Origin"].map(
        {1: "USA", 2: "Europe", 3: "Japan"},
    )
    y = dataset.pop("MPG")
    x = dataset.copy()

    return x, y

def make_train_test_split(x, y):

    from sklearn.model_selection import train_test_split

    (x_train, x_test, y_train, y_test) = train_test_split(
        x,
        y,
        test_size=0.25,
        random_state=123456,
    )
    return x_train, x_test, y_train, y_test

def make_pipeline(estimator):

    from sklearn.compose import ColumnTransformer
    from sklearn.feature_selection import SelectKBest, f_regression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    transformer = ColumnTransformer(
        transformers=[
            ("ohe", OneHotEncoder(dtype="int"), ["Origin"]),
        ],
        remainder=StandardScaler(),
    )

    selectkbest = SelectKBest(score_func=f_regression)

    pipeline = Pipeline(
        steps=[
            ("tranformer", transformer),
            ("selectkbest", selectkbest),
            ("estimator", estimator),
        ],
        verbose=False,
    )

    return pipeline

def make_grid_search(estimator, param_grid, cv=5):

    from sklearn.model_selection import GridSearchCV

    grid_search = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        cv=cv,
        scoring='neg_mean_absolute_error',
    )

    return grid_search

def save_estimator(estimator):

    import pickle

    with open("estimator.pickle", "wb") as file:
        pickle.dump(estimator, file)

def load_estimator():

    import os
    import pickle

    if not os.path.exists("estimator.pickle"):
        return None
    with open("estimator.pickle", "rb") as file:
        estimator = pickle.load(file)

    return estimator

def train_linear_regression():

    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_absolute_error
    

    data, target = load_data()

    x_train, x_test, y_train, y_test = make_train_test_split(
        x=data,
        y=target,
    )

    pipeline = make_pipeline(
        estimator=LinearRegression(),
    )

    estimator = make_grid_search(
        estimator=pipeline,
        param_grid={
            "selectkbest__k": range(1, len(x_train.columns) + 1),
        },
        cv=5,
    )

    estimator.fit(x_train, y_train)

    best_estimator = load_estimator()

    if best_estimator is not None:

        saved_mae = mean_absolute_error(
            y_true=y_test, y_pred=best_estimator.predict(x_test)
        )

        current_mae = mean_absolute_error(
            y_true=y_test, y_pred=estimator.predict(x_test)
        )

        if saved_mae < current_mae:
            estimator = best_estimator

    save_estimator(estimator)


train_linear_regression()

def eval_metrics(y_true, y_pred):

    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return mse, mae, r2

def report(estimator, mse, mae, r2):

    print(estimator, ":", sep="")
    print(f"  MSE: {mse}")
    print(f"  MAE: {mae}")
    print(f"   R2: {r2}")

def check_estimator():

    import pickle

    import pandas as pd
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    data, target = load_data()

    x_train, x_test, y_train_true, y_test_true = make_train_test_split(
        x=data,
        y=target,
    )

    estimator = load_estimator()

    mse, mae, r2 = eval_metrics(
        y_test_true,
        estimator.predict(x_test),
    )

    report(estimator.best_estimator_, mse, mae, r2)


check_estimator()

def train_mlp_regressor():

    from sklearn.neural_network import MLPRegressor
    from sklearn.metrics import mean_absolute_error

    data, target = load_data()

    x_train, x_test, y_train, y_test = make_train_test_split(
        x=data,
        y=target,
    )

    pipeline = make_pipeline(
        estimator=MLPRegressor(max_iter=30000),
    )

    estimator = make_grid_search(
        estimator=pipeline,
        param_grid={
            "selectkbest__k": range(1, len(x_train.columns) + 1),
            "estimator__hidden_layer_sizes": [(n,) for n in range(1, 11)],
            "estimator__solver": ["adam"],
            "estimator__learning_rate_init": [0.01, 0.001, 0.0001],
        },
        cv=5,
    )

    estimator.fit(x_train, y_train)

    best_estimator = load_estimator()

    if best_estimator is not None:

        saved_mae = mean_absolute_error(
            y_true=y_test, y_pred=best_estimator.predict(x_test)
        )

        current_mae = mean_absolute_error(
            y_true=y_test, y_pred=estimator.predict(x_test)
        )

        if saved_mae < current_mae:
            estimator = best_estimator

    save_estimator(estimator)


train_mlp_regressor()
check_estimator()

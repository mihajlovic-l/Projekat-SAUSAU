import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import GridSearchCV
from Projekat.Visualization import dt_rf_gb_metric
from sklearn.metrics import r2_score

def decision_tree(X_train, X_test, y_train, y_test, figs):
    #podesavanje hiperparametara i unakrsna validacija
    #skup vrednosti hiperparametara koje cemo testirati
    max_depths = [None]
    max_depths.extend(np.linspace(5, 50, 6, dtype=int))
    min_samples_splits = np.linspace(2, 100, 6, dtype=int)
    ccp_alphas = np.concatenate(([0], np.logspace(-4, 0, 6)))
    #print(ccp_alphas)

    param_grid_dt = {
        "max_depth": max_depths,
        "min_samples_split": min_samples_splits,
        "ccp_alpha": ccp_alphas,
        "criterion": ["squared_error", "poisson"],
    }

    all_dt_models = GridSearchCV(DecisionTreeRegressor(random_state=208), param_grid=param_grid_dt,
                                 scoring='neg_mean_squared_error', cv=5)
    all_dt_models.fit(X_train, y_train)

    #najbolji model i njegovi podaci
    best_dt = all_dt_models.best_estimator_
    best_params = all_dt_models.best_params_
    importances = best_dt.feature_importances_
    #print(best_params)
    y_pred_dt = best_dt.predict(X_test)

    #priprema podataka za vizualizaciju zavinosti ccp_alpha i depths/r2
    train_scores = []
    test_scores = []
    depths = []
    for ccp_alpha in ccp_alphas:
        model = DecisionTreeRegressor(random_state=208, ccp_alpha=ccp_alpha)
        model.fit(X_train, y_train)

        depths.append(model.tree_.max_depth)
        train_scores.append(r2_score(y_train, model.predict(X_train)))
        test_scores.append(r2_score(y_test, model.predict(X_test)))

    dt_rf_gb_metric(best_dt, "Decision Tree", y_test, y_pred_dt, best_params, importances,
                    ccp_alphas, depths, train_scores, test_scores, figs=figs)

def random_forest(X_train, X_test, y_train, y_test, figs):
    # podesavanje hiperparametara i unakrsna validacija
    param_grid_rf = {
        "n_estimators": [100, 300, 500],
        "max_depth": [None, 10, 20],
        "max_features": ["sqrt", "log2"]
    }
    all_rf_models = GridSearchCV(RandomForestRegressor(random_state=208), param_grid=param_grid_rf,
                                 scoring='neg_mean_squared_error', cv=5)
    all_rf_models.fit(X_train, y_train)

    # najbolji model i njegovi podaci
    best_rf = all_rf_models.best_estimator_
    best_params = all_rf_models.best_params_
    importances = best_rf.feature_importances_
    # print(best_params)
    y_pred_dt = best_rf.predict(X_test)

    #priprema podataka za vizualizaciju zavinosti n_estimators i r2
    train_scores = []
    test_scores = []
    n_estimators_list = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700]
    for n_estimators in n_estimators_list:
        model = RandomForestRegressor(random_state=208, n_estimators=n_estimators)
        model.fit(X_train, y_train)

        train_scores.append(r2_score(y_train, model.predict(X_train)))
        test_scores.append(r2_score(y_test, model.predict(X_test)))

    dt_rf_gb_metric(best_rf, "Random Forest", y_test, y_pred_dt, best_params, importances,
                    train_scores=train_scores, test_scores=test_scores, n_estimators=n_estimators_list, figs=figs)

def gradient_boosting(X_train, X_test, y_train, y_test, figs):
    # podesavanje hiperparametara i unakrsna validacija
    # provera da li se radi o proceni jednog parametra ili vise parametara
    # posto, za razliku od dt i rf, gb nije prilagodjen predvidanju vise izlaza, pa moramo da ga wrap-ujemo
    if y_train.ndim > 1:
        param_grid_gb = {
            "estimator__n_estimators": [100, 200, 500],
            "estimator__learning_rate": [0.01, 0.05, 0.1],
            "estimator__max_depth": [3, 5, 7]
        }
        all_gb_models = GridSearchCV(MultiOutputRegressor(GradientBoostingRegressor(random_state=208)), param_grid=param_grid_gb,
                                     scoring='neg_mean_squared_error', cv=5)
    else:
        param_grid_gb = {
            "n_estimators": [100, 200, 500],
            "learning_rate": [0.01, 0.05, 0.1],
            "max_depth": [3, 5, 7]
        }
        all_gb_models = GridSearchCV(GradientBoostingRegressor(random_state=208), param_grid=param_grid_gb,
                                     scoring='neg_mean_squared_error', cv=5)


    all_gb_models.fit(X_train, y_train)

    # najbolji model i njegovi parametri
    best_gb = all_gb_models.best_estimator_
    best_params = all_gb_models.best_params_
    if y_train.ndim > 1:
        importances = np.mean([est.feature_importances_ for est in best_gb.estimators_], axis=0)
    else:
        importances = best_gb.feature_importances_
    # print(best_params)
    y_pred_dt = best_gb.predict(X_test)

    # priprema podataka za vizualizaciju zavinosti n_estimators i r2, learning_rate i r2
    train_scores_n = []
    test_scores_n = []
    train_scores_lr = []
    test_scores_lr = []
    n_estimators_list = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700]
    learning_rate_list = np.logspace(-3, 0, 20)
    for n_estimators in n_estimators_list:
        if y_train.ndim > 1:
            model = MultiOutputRegressor(GradientBoostingRegressor(random_state=208, n_estimators=n_estimators))
        else:
            model = GradientBoostingRegressor(random_state=208, n_estimators=n_estimators)

        model.fit(X_train, y_train)

        train_scores_n.append(r2_score(y_train, model.predict(X_train)))
        test_scores_n.append(r2_score(y_test, model.predict(X_test)))

    for learning_rate in learning_rate_list:
        if y_train.ndim > 1:
            model = MultiOutputRegressor(GradientBoostingRegressor(random_state=208, learning_rate=learning_rate))
        else:
            model = GradientBoostingRegressor(random_state=208, learning_rate=learning_rate)

        model.fit(X_train, y_train)

        train_scores_lr.append(r2_score(y_train, model.predict(X_train)))
        test_scores_lr.append(r2_score(y_test, model.predict(X_test)))

    train_scores = [train_scores_n, train_scores_lr]
    test_scores = [test_scores_n, test_scores_lr]

    dt_rf_gb_metric(best_gb, "Gradient Boosting", y_test, y_pred_dt, best_params, importances,
                    train_scores=train_scores, test_scores=test_scores, n_estimators=n_estimators_list, learning_rates=learning_rate_list, figs=figs)
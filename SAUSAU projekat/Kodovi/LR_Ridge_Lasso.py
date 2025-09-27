from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.model_selection import GridSearchCV, cross_val_score
from Projekat.Visualization import lr_ridge_lasso_metric
from sklearn.multioutput import MultiOutputRegressor

def linear_regression(X_train, X_test, y_train, y_test):
    #provera da li se radi o proceni jednog parametra ili vise parametara
    if y_train.ndim > 1:
        lr_model = MultiOutputRegressor(LinearRegression())
    else:
        lr_model = LinearRegression()

    #unakrsna validacija samo radi provere
    lr_cv_scores = cross_val_score(lr_model, X_train, y_train, scoring="neg_mean_squared_error", cv=5)

    lr_model.fit(X_train, y_train)
    y_pred_lr = lr_model.predict(X_test)

    lr_ridge_lasso_metric(lr_model, "Linear", y_test, y_pred_lr, lr_cv_scores=lr_cv_scores)

def ridge(X_train, X_test, y_train, y_test):
    #podesavanje hiperparametara (alpha) i unakrsna validacija
    #provera da li se radi o proceni jednog parametra ili vise parametara
    if y_train.ndim > 1:
        alpha_grid_ridge = {'estimator__alpha': [0.01, 0.1, 1, 10, 100]}
        all_ridge_models = GridSearchCV(MultiOutputRegressor(Ridge()), alpha_grid_ridge, scoring='neg_mean_squared_error', cv=5)
    else:
        alpha_grid_ridge = {'alpha': [0.01, 0.1, 1, 10, 100]}
        all_ridge_models = GridSearchCV(Ridge(), alpha_grid_ridge,scoring='neg_mean_squared_error', cv=5)

    all_ridge_models.fit(X_train, y_train)

    #odabir modela sa najboljim alpha
    best_ridge = all_ridge_models.best_estimator_
    if y_train.ndim > 1:
        best_alpha = all_ridge_models.best_params_['estimator__alpha']
    else:
        best_alpha = all_ridge_models.best_params_['alpha']
    y_pred_ridge = best_ridge.predict(X_test)

    lr_ridge_lasso_metric(best_ridge, "Ridge", y_test, y_pred_ridge, alpha=best_alpha)

def lasso(X_train, X_test, y_train, y_test):
    #podesavanje hiperparametara (alpha) i unakrsna validacija
    #provera da li se radi o proceni jednog parametra ili vise parametara
    if y_train.ndim > 1:
        alpha_grid_lasso = {'estimator__alpha': [0.0001, 0.001, 0.01, 0.1, 1]}
        all_lasso_models = GridSearchCV(MultiOutputRegressor(Lasso(max_iter=10000)), alpha_grid_lasso, scoring='neg_mean_squared_error', cv=5)
    else:
        alpha_grid_lasso = {'alpha': [0.0001, 0.001, 0.01, 0.1, 1]}
        all_lasso_models = GridSearchCV(Lasso(max_iter=10000), alpha_grid_lasso, scoring='neg_mean_squared_error', cv=5)

    all_lasso_models.fit(X_train, y_train)

    #odabir modela sa najboljim alpha
    best_lasso = all_lasso_models.best_estimator_
    if y_train.ndim > 1:
        best_alpha = all_lasso_models.best_params_['estimator__alpha']
    else:
        best_alpha = all_lasso_models.best_params_['alpha']
    y_pred_lasso = best_lasso.predict(X_test)

    lr_ridge_lasso_metric(best_lasso, "Lasso", y_test, y_pred_lasso, alpha=best_alpha)
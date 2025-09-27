from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error,root_mean_squared_error
from sklearn.tree import plot_tree

def lr_ridge_lasso_metric(model, model_name, y_test, y_pred, lr_cv_scores = None, alpha=0):
    mse = mean_squared_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)


    print(f"\n---------------------------------------{model_name}---------------------------------------\n")
    if model_name == "Linear":
        print("CV MSEs:", -lr_cv_scores)
    if model_name == "Ridge" or model_name == "Lasso":
        print("Best alpha:", alpha)
    if y_pred.ndim == 1:
        coef = model.coef_
        print("Coefficients:", coef)
    print("MSE:", mse)
    print("RMSE:", rmse)
    print("MAE:", mae)
    print("R2 Score:", r2)
    print()

def dt_rf_gb_metric(model, model_name, y_test, y_pred, best_params, importances,
                    ccp_alphas = None, depths = None, train_scores = None, test_scores = None, n_estimators = None, learning_rates = None, figs = None):

    mode = "Multi-output" if y_pred.ndim > 1 else "Single output"

    best_params_clean = {k: v.item() if hasattr(v, "item") else v
                   for k, v in best_params.items()}      #konvertuje np.float(broj) i np.int(broj) u broj

    mse = mean_squared_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"\n---------------------------------------{model_name}---------------------------------------\n")
    print("Best parameters:", best_params_clean)
    print("Feature importances:")
    for name, importance in zip(model.feature_names_in_, importances):
        print(f"{name}: {importance:.4f}")
    print()
    print("MSE:", mse)
    print("RMSE:", rmse)
    print("MAE:", mae)
    print("R2 Score:", r2)
    print()

    if figs is None:
        figs = []

    if model_name == "Decision Tree":
        fig, ax = plt.subplots(2, 1, figsize=(8, 10))

        # 1st subplot: alpha vs depth
        ax[0].plot(ccp_alphas, depths, marker="o")
        ax[0].set_xlabel("ccp_alpha")
        ax[0].set_ylabel("Depth")
        ax[0].set_title(f"{mode} Decision Tree ccp_alpha vs Depth")

        # 2nd subplot: alpha vs r2
        ax[1].plot(ccp_alphas, train_scores, marker="o", label="Train R²")
        ax[1].plot(ccp_alphas, test_scores, marker="o", label="Test R²")
        ax[1].set_xlabel("ccp_alpha")
        ax[1].set_ylabel("R²")
        ax[1].set_title(f"{mode} Decision Tree ccp_alpha vs R²")
        ax[1].legend()

        plt.tight_layout()

        figs.append(fig)

        fig, ax = plt.subplots(figsize=(20, 10))
        plot_tree(
            model,
            filled=True,
            feature_names=model.feature_names_in_,
            rounded=True,
            max_depth=3
        )
        plt.title(f"{mode} Decision Tree (max depth = 3)")
        figs.append(fig)
    elif model_name == "Random Forest":
        fig, ax = plt.subplots(figsize=(8, 5))
        plt.plot(n_estimators, train_scores, marker="o", label="Train R²")
        plt.plot(n_estimators, test_scores, marker="o", label="Test R²")
        plt.xlabel("n_estimators")
        plt.ylabel("R² Score")
        plt.title(f"{mode} Random Forest: n_estimators vs. R²")
        plt.legend()
        plt.grid(True)
        figs.append(fig)
    else:
        fig, ax = plt.subplots(2, 1, figsize=(8, 5))

        # 1st subplot: n_estimators vs r2
        ax[0].plot(n_estimators, train_scores[0], marker="o", label="Train R²")
        ax[0].plot(n_estimators, test_scores[0], marker="o", label="Test R²")
        ax[0].set_xlabel("n_estimators")
        ax[0].set_ylabel("R²")
        ax[0].set_title(f"{mode} Gradient Boosting n_estimators vs R²")
        ax[0].legend()

        # 2nd subplot: learning_rate vs r2
        ax[1].plot(learning_rates, train_scores[1], marker="o", label="Train R²")
        ax[1].plot(learning_rates, test_scores[1], marker="o", label="Test R²")
        ax[1].set_xlabel("learning_rates")
        ax[1].set_ylabel("R²")
        ax[1].set_title(f"{mode} Gradient Boosting learning_rates vs R²")
        ax[1].legend()

        plt.tight_layout()

        figs.append(fig)
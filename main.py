import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import GridSearchCV
import logging

# 日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# 数据预处理
def load_and_preprocess_data(filepath):
    logging.info("Loading data from: {}".format(filepath))
    df = pd.read_csv(filepath)
    logging.info("Data loaded successfully. Shape: {}".format(df.shape))

    df.dropna(inplace=True)  # 处理缺失值
    logging.info("Missing values handled. New shape: {}".format(df.shape))

    X = df.drop(columns=['target'])  # 假设目标变量列名为'target'
    y = df['target']

    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    logging.info("Data standardization completed.")

    return X_scaled, y


# 特征
def feature_selection(X, y, k=10):
    logging.info(f"Performing feature selection. Selecting top {k} features.")
    selector = SelectKBest(f_regression, k=k)
    X_new = selector.fit_transform(X, y)
    selected_features = selector.get_support(indices=True)
    logging.info(f"Selected features: {selected_features}")
    return X_new, selected_features


# 随机森林回归
def build_random_forest_model(X_train, y_train):
    logging.info("Building Random Forest model...")
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    logging.info("Random Forest model trained successfully.")
    return rf_model


# XGBoost回归
def build_xgboost_model(X_train, y_train):
    logging.info("Building XGBoost model...")
    xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.01, max_depth=6)
    xgb_model.fit(X_train, y_train)
    logging.info("XGBoost model trained successfully.")
    return xgb_model


# 模型评估
def evaluate_model(model, X_test, y_test):
    logging.info(f"Evaluating model: {model}")
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    logging.info(f"Mean Squared Error: {mse}")
    logging.info(f"R2 Score: {r2}")
    return mse, r2


# 模型调参
def tune_model(model, X_train, y_train):
    logging.info(f"Tuning model using GridSearchCV...")
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7, 10],
        'learning_rate': [0.001, 0.01, 0.1]
    }
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error',
                               n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    logging.info(f"Best parameters found: {grid_search.best_params_}")
    return best_model


# 集成预测
def ensemble_predict(rf_model, xgb_model, X_test):
    logging.info("Generating ensemble prediction...")
    rf_pred = rf_model.predict(X_test)
    xgb_pred = xgb_model.predict(X_test)
    ensemble_pred = (rf_pred + xgb_pred) / 2
    return ensemble_pred


# 交叉验证
def cross_validation(model, X_train, y_train, cv_folds=5):
    logging.info("Performing cross-validation...")
    scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring='neg_mean_squared_error')
    logging.info(f"Cross-validation scores: {scores}")
    return scores


# 可视化结果
def visualize_results(y_test, ensemble_pred, rf_pred, xgb_pred):
    logging.info("Visualizing results...")
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.regplot(x=y_test, y=ensemble_pred, line_kws={'color': 'red'})
    plt.title('Ensemble Model Prediction vs Actual')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')

    plt.subplot(1, 2, 2)
    sns.regplot(x=y_test, y=rf_pred, line_kws={'color': 'blue'})
    sns.regplot(x=y_test, y=xgb_pred, line_kws={'color': 'green'})
    plt.title('Random Forest and XGBoost Predictions vs Actual')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.tight_layout()
    plt.show()


# 主函数
def main():
    logging.info("Starting the model training process.")
    # 加载数据
    X, y = load_and_preprocess_data('battery_performance_data.csv')

    # 特征选择
    X_selected, selected_features = feature_selection(X, y, k=10)

    # 切分数据
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

    # 构建基础模型
    rf_model = build_random_forest_model(X_train, y_train)
    xgb_model = build_xgboost_model(X_train, y_train)

    # 模型调参
    rf_model_tuned = tune_model(rf_model, X_train, y_train)
    xgb_model_tuned = tune_model(xgb_model, X_train, y_train)

    # 模型评估
    rf_mse, rf_r2 = evaluate_model(rf_model_tuned, X_test, y_test)
    xgb_mse, xgb_r2 = evaluate_model(xgb_model_tuned, X_test, y_test)

    # 集成预测
    ensemble_pred = ensemble_predict(rf_model_tuned, xgb_model_tuned, X_test)
    ensemble_mse = mean_squared_error(y_test, ensemble_pred)
    logging.info(f"Ensemble Model Mean Squared Error: {ensemble_mse}")

    # 交叉验证
    rf_cv_scores = cross_validation(rf_model_tuned, X_train, y_train)
    xgb_cv_scores = cross_validation(xgb_model_tuned, X_train, y_train)

    # 可视化结果
    visualize_results(y_test, ensemble_pred, rf_model_tuned.predict(X_test), xgb_model_tuned.predict(X_test))

    logging.info("Model training and evaluation completed.")


if __name__ == '__main__':
    main()
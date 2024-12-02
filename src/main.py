import tensorflow as tf

from load_data import DataLoader
from model import Model
from visualization import Visualize

# Function which handles the linear regression model
def linearRegression(X_train, Y_train, X_test, Y_test, visualize):
    lr_model = Model(model_type='linear')
    lr_model.train(X_train, Y_train)

    mean_score, stf_dev = lr_model.cross_validate(lr_model.model, X_train, Y_train)
    mean, r2 = lr_model.evaluate(X_test, Y_test)

    print("\033[1;4mLinear Regression Evaluation\033[0m\n")
    print(f"R2 Score: {r2} \nMean Score: {mean}")
    print(f"Cross Validation Score: {mean_score} \nStandard Deviation: {stf_dev}")
    print("\n")

    visualize.plot_scatter(Y_test, lr_model.predict(X_test))

# Function which handles the Random Forest ensemble
def randomForest(X_train, Y_train, X_test, Y_test, visualize):
    rf_model = Model(model_type='random_forest')
    param_grid = {
        'n_estimators': [100,200, 300],
        'max_depth': [None, 10, 20, 30],
        'max_features': ['sqrt', 'log2', None]
    }
    best_params = rf_model.tune_hyperparameters(X_train, Y_train, param_grid)
    rf_model = Model(model_type='random_forest', params=best_params)
    rf_model.train(
        X_train, Y_train
    )

    mean_score, stf_dev = rf_model.cross_validate(rf_model.model, X_train, Y_train)
    mean, r2 = rf_model.evaluate(X_test, Y_test)

    print("\033[1;4mRandom Forest Evaluation\033[0m\n")
    print(f"R2 Score: {r2} \nMean Score: {mean}")
    print(f"Cross Validation Score: {mean_score} \nStandard Deviation: {stf_dev}")
    print("\n")
    visualize.plot_scatter(Y_test, rf_model.predict(X_test))
    #feature_names = X_train.columns.tolist()
    #visualize.plot_importance(rf_model.model, feature_names)

def xgBoost(X_train, Y_train, X_test, Y_test, visualize):
    xgb_model = Model(model_type='xgboost')
    xgb_model.train(X_train, Y_train)
    xgb_evaluation = xgb_model.evaluate(X_test, Y_test)
    print("XGBoost Evaluation: ", xgb_evaluation)

    visualize.plot_scatter(Y_test, xgb_model.predict(X_test))
    feature_names = X_train.columns.tolist()
    visualize.plot_importance(xgb_model.model, feature_names)

def neuralNetwork(X_train, Y_train, X_test, Y_test, visualize):


    print("\033[1;4mNeural Network Evaluation\033[0m\n")
    print(f"R2 Score: {r2} \nMean Score: {mean}")
    print(f"Cross Validation Score: {mean_score} \nStandard Deviation: {stf_dev}")
    print("\n")
    visualize.plot_scatter(Y_test, sq.predict(X_test))

def main():

    filepath = '../data/data.csv'
    target = 'price'

    ld = DataLoader(filepath, target)
    ld.load_data()
    ld.preprocess_data()
    ld.split_data()

    X_train, X_test, Y_train, Y_test = ld.get_data()

    viz = Visualize()

    linearRegression(X_train, Y_train, X_test, Y_test, viz)
    neuralNetwork(X_train, Y_train, X_test, Y_test, viz)
    # randomForest(X_train, Y_train, X_test, Y_test, viz)
    # xgBoost(X_train, Y_train, X_test, Y_test, viz)

if __name__ == "__main__":
    main()


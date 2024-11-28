from load_data import DataLoader
from model import Model
from visualization import Visualize

def linearRegression(X_train, Y_train, X_test, Y_test, visualize):
    lr_model = Model(model_type='linear')
    lr_model.train(X_train, Y_train)
    lr_evaluation = lr_model.evaluate(X_test, Y_test)
    print("Linear Regression Evaluation: ", lr_evaluation)

    visualize.plot_scatter(Y_test, lr_model.predict(X_test))

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
    rf_evaluation = rf_model.evaluate(X_test, Y_test)
    print("Random Forest Evaluation: ", rf_evaluation)

    visualize.plot_scatter(Y_test, rf_model.predict(X_test))

def xgBoost(X_train, Y_train, X_test, Y_test):
    xgb_model = Model(model_type='xgboost')
    xgb_model.train(X_train, Y_train)
    xgb_evaluation = xgb_model.evaluate(X_test, Y_test)
    print("XGBoost Evaluation: ", xgb_evaluation)

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
    randomForest(X_train, Y_train, X_test, Y_test, viz)
    xgBoost(X_train, Y_train, X_test, Y_test)

if __name__ == "__main__":
    main()


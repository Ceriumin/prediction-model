from load_data import DataLoader
from model import Model
from visualization import Visualize
import time

# Function which handles the linear regression model
def linearRegression(X_train, Y_train, X_test, Y_test, visualize):
    name = "Linear Regression"
    model = Model(model_type='linear')

    model.train(X_train, Y_train)
    model.evaluate(X_test, Y_test, name)
    visualize.plot_scatter(Y_test, model.predict(X_test), name)

def randomForest(X_train, Y_train, X_test, Y_test, visualize):
    name = "Random Forest"
    model = Model(model_type='random_forest')
    param_grid = {
        'n_estimators': [100,200, 300],
        'max_depth': [None, 10, 20, 30],
        'max_features': ['sqrt', 'log2', None]
    }
    best_params = model.tune_hyperparameters(X_train, Y_train, param_grid)
    tuned = Model(model_type='random_forest', params=best_params)
    tuned.train(X_train, Y_train)

    model.evaluate(X_test, Y_test, name)
    # visualize.plot_scatter(Y_test, rf_model.predict(X_test))
    #feature_names = X_train.columns.tolist()
    #visualize.plot_importance(rf_model.model, feature_names)

def xgBoost(X_train, Y_train, X_test, Y_test, visualize):
    name = "XGBoost"
    model = Model(model_type='xgboost')
    model.train(X_train, Y_train)

    model.evaluate(X_test, Y_test, name)
    visualize.plot_scatter(Y_test, model.predict(X_test))
    # feature_names = X_train.columns.tolist()
    # visualize.plot_importance(xgb_model.model, feature_names)

def combined(X_train, Y_train, X_test, Y_test, visualize):
    name = "Stacking Ensemble"
    model = Model(model_type='all')
    model.train(X_train, Y_train)

    model.evaluate(X_test, Y_test, name)
    visualize.plot_scatter(Y_test, model.predict(X_test))

def keras(X_train, Y_train, X_test, Y_test, visualize):
    name = "Neural Network"
    model = Model(model_type='keras')
    model.model.fit(X_train, Y_train, epochs=100, batch_size=32, validation_data=(X_test, Y_test), verbose=0)

    model.evaluate(X_test, Y_test, name)
    visualize.plot_scatter(Y_test, model.predict(X_test))


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
    # combined(X_train, Y_train, X_test, Y_test, viz)
    # keras(X_train, Y_train, X_test, Y_test, viz)
    randomForest(X_train, Y_train, X_test, Y_test, viz)
    # xgBoost(X_train, Y_train, X_test, Y_test, viz)

if __name__ == "__main__":
    main()


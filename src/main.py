from load_data import DataLoader
from model import Model
from visualization import Visualize

# Function which handles the linear regression model
def linearRegression(X_train, Y_train, X_test, Y_test, visualize):
    lr_model = Model(model_type='linear')
    lr_model.train(X_train, Y_train)

    mean, stf_dev = lr_model.cross_validate(lr_model.model, X_train, Y_train)
    mse, r2, mae = lr_model.evaluate(X_test, Y_test)

    print("\033[1;4mLinear Regression Evaluation\033[0m\n")
    print(f"R2 Score: {r2} \nMean Squared Error: {mse}\nMean Absolute Error: {mae}")
    print(f"Cross Validation Score: {mean} \nStandard Deviation: {stf_dev}")
    print("\n")

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
    rf_model.train(X_train, Y_train)

    mean, stf_dev = rf_model.cross_validate(rf_model.model, X_train, Y_train)
    mse, r2, mae = rf_model.evaluate(X_test, Y_test)

    print("\033[1;4mRandom Forest Evaluation\033[0m\n")
    print(f"R2 Score: {r2} \nMean Squared Error: {mse}\nMean Absolute Error: {mae}")
    print(f"Cross Validation Score: {mean} \nStandard Deviation: {stf_dev}")
    print("\n")
    visualize.plot_scatter(Y_test, rf_model.predict(X_test))
    #feature_names = X_train.columns.tolist()
    #visualize.plot_importance(rf_model.model, feature_names)

def xgBoost(X_train, Y_train, X_test, Y_test, visualize):
    xgb_model = Model(model_type='xgboost')
    xgb_model.train(X_train, Y_train)

    mean, stf_dev = xgb_model.cross_validate(xgb_model.model, X_train, Y_train)
    mse, r2, mae = xgb_model.evaluate(X_test, Y_test)

    print("\033[1;4mXGBoost Evaluation\033[0m\n")
    print(f"R2 Score: {r2} \nMean Squared Error: {mse}\nMean Absolute Error: {mae}")
    print(f"Cross Validation Score: {mean} \nStandard Deviation: {stf_dev}")
    print("\n")

    visualize.plot_scatter(Y_test, xgb_model.predict(X_test))
    # feature_names = X_train.columns.tolist()
    # visualize.plot_importance(xgb_model.model, feature_names)

def combined(X_train, Y_train, X_test, Y_test, visualize):
    meta = Model(model_type='all')
    meta.train(X_train, Y_train)

    mean, stf_dev = meta.cross_validate(meta.model, X_train, Y_train)
    mse, r2, mae = meta.evaluate(X_test, Y_test)

    print("\033[1;4mCombined Evaluation\033[0m\n")
    print(f"R2 Score: {r2} \nMean Squared Error: {mse}\nMean Absolute Error: {mae}")
    print(f"Cross Validation Score: {mean} \nStandard Deviation: {stf_dev}")
    print("\n")

    visualize.plot_scatter(Y_test, meta.predict(X_test))

def keras(X_train, Y_train, X_test, Y_test, visualize):
    keras_model = Model(model_type='keras')
    keras_model.model.fit(X_train, Y_train, epochs=100, batch_size=32, validation_data=(X_test, Y_test), verbose=0)
    mse, r2, mae = keras_model.evaluate(X_test, Y_test)

    print("\033[1;4mKeras Evaluation\033[0m\n")
    print(f"R2 Score: {r2} \nMean Squared Error: {mse}\nMean Absolute Error: {mae}")
    print("\n")

    visualize.plot_scatter(Y_test, keras_model.predict(X_test))


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
    combined(X_train, Y_train, X_test, Y_test, viz)
    keras(X_train, Y_train, X_test, Y_test, viz)
    randomForest(X_train, Y_train, X_test, Y_test, viz)
    xgBoost(X_train, Y_train, X_test, Y_test, viz)

if __name__ == "__main__":
    main()


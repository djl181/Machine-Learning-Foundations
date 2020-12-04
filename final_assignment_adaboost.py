import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier


def about(dataframe):

    df = dataframe.copy(deep=True)
    print('=' * 30)

    print('Sample data (Top 10 records')
    print(df.head(10))
    print('=' * 30)

    print('Data Shape')
    print(df.shape)
    print('=' * 30)

    print(df.describe())
    print('=' * 30)

    print('Feature Means at Quality = 8 (good)')
    print('=' * 30)
    sorted_df = df.sort_values(by='quality', ascending=False)
    print(df[df['quality'] == 8].mean())
    print('=' * 30)

    print('Correlation')
    print('=' * 30)
    corr_matrix = df.corr()
    print(corr_matrix["quality"].sort_values(ascending=False))
    print('=' * 30)

    bins = (0, 6.5, 10)
    group_names = ['poor (1)', 'acceptable (0)']
    df['quality'] = pd.cut(df['quality'], bins=bins, labels=group_names)
    # sns.displot(df['quality'], kde=True, bins='auto', color='darkblue')
    print('Quality Counts')
    print(df['quality'].value_counts())
    print('=' * 30)

    label_quality = LabelEncoder()
    # Bad becomes 1 and good becomes 0
    df['quality'] = label_quality.fit_transform(df['quality'])
    print('Quality Counts - Binary Target')
    print(df['quality'].value_counts())
    print('=' * 30)


def adaboost_classifier(df, iterations=10, minimum_error=1e-6, verbose=False):

    adaboost_model = {
                    "weights": [],
                    "weak predictors": [],
                    "iterations": iterations,
                    "minimum error": minimum_error
    }

    # Create Data Sets X and y
    X = df.drop('quality', axis=1)
    y = df['quality']

    #  Step 1. Initialise Hypothesis
    n = len(df)
    weights = np.ones(n) / n

    # Step 2. Create loop for the number of boosting iterations
    iter_num = 0
    while True:

        # Step 3. resample the data
        # Use a random index set, to sample both the X- and y-array and need to keep the (X-y) correspondence consistent
        df_resampled = df.sample(frac=1, replace=True, weights=weights, axis=0)

        # add an extra dimension (axis) to satisfy the argument DecisionTree model functions `predict` and `fit`
        # they want the data array to be [#.samples x #.attributes] #.attributes == 1
        x_resampled_train = df_resampled.drop('quality', axis=1)
        y_resampled_train = df_resampled['quality']

        # Step 4. Fit weak classifier using resampled data
        weak_classifier = DecisionTreeClassifier(max_depth=1)
        weak_classifier.fit(x_resampled_train, y_resampled_train)

        # Step 5. Apply Hypothesis to data and calculate Hypothesis Error
        h_error = (weak_classifier.predict(x_resampled_train) != y_resampled_train).astype(float)

        # Step 6. Calculate the weighted error
        w_error = max(np.sum(h_error) / n, minimum_error)  # Ensure the following quotient is numerically safely defined

        # Step 7. Calculate the weight (alpha) for this weak classifier
        alpha = 1 / 2 * np.log((1 - w_error) / w_error)

        # Step 8  Update Weights
        # err_0 = (weak_classifier.predict(X[:, np.newaxis]) != y).astype(float)
        err_0 = (weak_classifier.predict(X) != y).astype(float)
        dw = np.exp(alpha * (err_0 - 0.5) * 2)
        weights = dw * weights
        weights = weights / weights.sum()

        # Store the information of the model for this boosting iteration
        adaboost_model["weights"].append(alpha)
        adaboost_model["weak predictors"].append(weak_classifier)

        # Report
        if verbose:
            print("\n========")
            print(f"\nIterations: {iter_num}")
            print("\nResampled_X_Train", x_resampled_train.squeeze())
            print("\nError", w_error)
            print("\nPredictor weight", alpha)
            print("\nSample weights", weights)

        # stop condition is maximum iterations or minimum error reached
        iter_num += 1
        if iter_num >= iterations or w_error <= minimum_error:
            break

    adaboost_model["resampled_X_train"] = x_resampled_train.squeeze()
    adaboost_model["error"] = w_error
    adaboost_model["predictor weight"] = alpha
    adaboost_model["sample weight"] = weights

    return adaboost_model


def accuracy(prediction, test_dataset):

    results = {}

    # Create Dataset y
    print('Test Data Quality Counts')
    y = np.array(test_dataset['quality'])
    results['Test Data Quality Counts'] = test_dataset['quality'].value_counts()
    print(results['Test Data Quality Counts'])
    print('=' * 30)


    print('my_adaboost_class')
    print('Model Performance')
    results['Model Average Error'] = mean_absolute_error(y, prediction)
    results['Model Accuracy'] = accuracy_score(y, prediction, normalize=True) * 100
    results['Model Quality Counts'] = pd.DataFrame(prediction).value_counts()

    print('Average Error: {:0.4f} degrees.'.format(results['Model Average Error']))
    print('Accuracy = {:0.2f}%.'.format(results['Model Accuracy']))
    print('=' * 30)
    print(results['Model Quality Counts'])
    print('=' * 30)


    return results


def box_plot(results):
    fig = plt.figure()
    fig.suptitle('Algorithm Results')
    ax = fig.add_subplot(111)
    plt.boxplot(results)
    ax.set_xticklabels(['class_adaboost_model'])

    fig.show()

    return


def classify(dataframe, limit, low=-1, high=1):

    # Dividing wine as good and bad by giving the limit for the quality,
    bins = (0, limit, 10)
    group_names = [low, high]
    dataframe['quality'] = pd.cut(dataframe['quality'], bins=bins, labels=group_names)
    sorted_df = dataframe.sort_values(by='quality', ascending=False)

    return sorted_df


def fit(dataframe, iterations=30, verbose=False):
    class_adaboost_model = adaboost_classifier(dataframe, iterations=iterations, verbose=verbose)

    print('adaboost model results')
    print('=' * 30)
    [print(f'{k}: {v}\n') for k, v in class_adaboost_model.items()]
    print('=' * 30)


    return class_adaboost_model


def multiple_runs(train, test, fit_iterations, iterations=10):

    # from csvfile3 import CsvFile

    total_results = []
    for n in range(iterations):
        print(f'Sample run: {n}')
        print('=' * 30)
        class_adaboost_model = fit(train, iterations=fit_iterations, verbose=False)
        prediction = predict(class_adaboost_model, test)
        total_results.append(accuracy(prediction['sign'], test))

    [print(_) for _ in total_results]

    # csv = CsvFile()
    # csv.header(['Test Data Quality Counts', 'Model Average Error', 'Model Accuracy', 'Model Quality Counts' ])
    # for _ in total_results:
    #     csv.append(row=_)

    # csv.write('results.csv')


def prepare(filename, threshold=6.5):

    pd.set_option('display.width', 2000)
    pd.set_option('display.max_columns', 12)

    df = pd.read_csv(filename, sep=';')
    about(df)

    # From data inspection the arbitrary figure shall be 6.5
    classified_df = classify(df, limit=threshold)

    return classified_df


def split(dataframe, test_size=0.2):

    train, test = train_test_split(dataframe, test_size=test_size, random_state=42, shuffle=True)

    return train, test


def predict(adaboost_model, dataset):

    # Create Data Sets X and y
    X = dataset.drop('quality', axis=1)
    y = dataset['quality']

    #
    total_prediction = np.zeros_like(y, dtype=float)
    for a, g in zip(adaboost_model['weights'], adaboost_model['weak predictors']):
        pred = g.predict(X)
        total_prediction += a * pred

    total_prediction_sign = np.sign(total_prediction)

    return {'value': total_prediction, 'sign': total_prediction_sign}


def main():

    filename = 'winequality-red.csv'
    target_binary_threshold = 6.5
    split_test_size = 0.2
    fit_iterations = 30

    # Load and prepare dataset
    prepared_df = prepare(filename, threshold=target_binary_threshold)

    # Split the data into train and test set
    train, test = split(prepared_df, test_size=split_test_size)

    # Fit the train data using the adaboost classifier
    class_adaboost_model = fit(train, iterations=fit_iterations, verbose=False)

    # Make prediction using Test dataset
    prediction = predict(class_adaboost_model, test)

    # Print accuracy
    accuracy(prediction['sign'], test)

    # Plot
    box_plot(prediction['value'])

    multiple_runs(train, test, fit_iterations, iterations=10)


if __name__ == '__main__':
    main()

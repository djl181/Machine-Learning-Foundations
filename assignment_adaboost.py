import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import  accuracy_score, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier


def my_adaboost(X_train, y_train, X_test, y_test, iterations):

    # Initialize weights
    n_train = len(X_train)
    n_test = len(X_test)
    w = np.ones(n_train) / n_train

    # Initialize predictions

    pred_train, pred_test = [np.zeros(n_train), np.zeros(n_test)]

    d_tree = DecisionTreeClassifier(max_depth=1, random_state=1)
    importance = []

    for i in range(iterations):
        # Fit a classifier with the specific weights
        d_tree.fit(X_train, y_train, sample_weight=w)
        importance.append(list( d_tree.feature_importances_))
        print(d_tree.feature_importances_)
        pred_train_i = d_tree.predict(X_train)
        pred_test_i = d_tree.predict(X_test)

        # Indicator function
        miss = [int(x) for x in (pred_train_i != y_train)]

        # Equivalent with 1/-1 to update weights
        miss2 = [x if x == 1 else -1 for x in miss]

        # Error
        err_m = np.dot(w, miss) / sum(w)

        # Alpha
        alpha_m = 0.5 * np.log((1 - err_m) / float(err_m))

        # New weights
        w = np.multiply(w, np.exp([float(x) * alpha_m for x in miss2]))

        # Add to prediction
        pred_train = [sum(x) for x in zip(pred_train, [x * alpha_m for x in pred_train_i])]
        pred_test = [sum(x) for x in zip(pred_test, [x * alpha_m for x in pred_test_i])]

    pred_train = np.sign(pred_train)
    pred_test = np.sign(pred_test)

    # Return error rate in train and test set
    er_train = sum(pred_train != y_train) / float(len(y_train))
    er_test = sum(pred_test != y_test) / float(len(y_test))

    return pred_train, er_train, pred_test, er_test, w, sum(np.array(importance))/iterations


def about(df):

    # print(df.head(10))
    # print(df.shape)
    print(df.describe())
    print('=' * 30)

    print('Mean Quality = 8')
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
    group_names = ['poor', 'acceptable']
    df['quality'] = pd.cut(df['quality'], bins=bins, labels=group_names)
    sns.displot(df['quality'], kde=True, bins='auto', color='darkblue')

def fit(X, y):
    d_tree = DecisionTreeClassifier(max_depth=1, random_state=1)
    d_tree.fit(X, y)

    return d_tree

def load_dataset(filename):

    df = pd.read_csv('winequality-red.csv', sep=';')

    return df

def main():
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_columns', 20)

    df = load_dataset('winequality-red.csv')

    # Fit a decision stump first
    d_tree = DecisionTreeClassifier(max_depth=1, random_state=1)
    d_tree.fit(X_train, y_train)

    pred_train = d_tree.predict(X_train)
    pred_test = d_tree.predict(X_test)

    er_train = sum(pred_train != y_train) / float(len(y_train))
    er_test = sum(pred_test != y_test) / float(len(y_test))

    # Fit my_adaboost using a decision tree as base estimator
    pred_train, er_train, pred_test, er_test, weights, importance = my_adaboost(X_train, y_train, X_test, y_test, 1000)
    print('=' * 30)
    print('my_adaboost Test')
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(mean_absolute_error(y_test, pred_test)))
    print('Accuracy = {:0.2f}%.'.format(accuracy_score(y_test, pred_test) * 100))

    # Plot the feature importance of the forest
    plt.figure()
    plt.title("myadaboost Feature importance")
    plt.barh(range(X.shape[1]), importance[indices], color="b",  align="center")


    colum_names = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']
    plt.yticks(range(X.shape[1]), colum_names)
    plt.ylim([0, X.shape[1]])
    plt.show()

def predict(d_tree, X):
    prediction = d_tree.predict(X)

    error = sum(pred_train != y_train) / float(len(y_train))
    er_test = sum(pred_test != y_test) / float(len(y_test))

    # Fit my_adaboost using a decision tree as base estimator
    pred_train, er_train, pred_test, er_test, weights, importance = my_adaboost(X_train, y_train, X_test, y_test, 1000)


def preprocess_data(df):

    bins = (0, 6.5, 10)
    group_names = ['poor', 'acceptable']
    df['quality'] = pd.cut(df['quality'], bins=bins, labels=group_names)

    print('=' * 30)
    print(df.head(40))
    print('=' * 30)
    print(df['quality'].value_counts())

    sorted_df = df.sort_values(by='quality', ascending=False)
    print(sorted_df['quality'])

    print('=' * 30)
    print('Quality Counts')
    label_quality = LabelEncoder()
    # Bad becomes 0 and good becomes 1
    df['quality'] = label_quality.fit_transform(df['quality'])
    print(df['quality'].value_counts())
    print('=' * 30)

    return df

def split(df):
    # Create Data Sets X and y
    X = df.drop('quality', axis=1)
    y = df['quality']

    # Standardise, Train and Test splitting of data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.fit_transform(X_test)
    print('=' * 30)

    return X, y, X_train, X_test, y_train, y_test

if __name__ == '__main__':
    main()

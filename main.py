import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle 
from sklearn.model_selection import train_test_split
from backprop import forward_prop, compute_gradients, update_weights, compute_final_cost

#WDBC DATASET
# data = pd.read_csv('data/wdbc.csv',header=0)

# X = data.iloc[:, :-1]
# Y = data.iloc[:, -1]

# min_value = X.min()
# max_value = X.max()
# X = (X-min_value) / (max_value-min_value)

# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

#-----------------------------------------------------------------------------

# LOAN DATASET
# df_loan = pd.read_csv("data/loan.csv")
# X = df_loan.drop(columns=["label"])
# Y = df_loan["label"]

# # one-hote encode categorical data 
# X = pd.get_dummies(X, columns=[
#     "attr1_cat", "attr2_cat", "attr3_cat", "attr4_cat", "attr5_cat",
#     "attr10_cat", "attr11_cat"
# ])

# # normalize numerical data
# num_cols = ["attr6_num", "attr7_num", "attr8_num", "attr9_num"]
# min_val = X[num_cols].min()
# max_val = X[num_cols].max()
# X[num_cols] = (X[num_cols] - min_val) / (max_val - min_val)
# X  = X.astype(np.float64)

#------------------------------------------------------------------------------

#RAISIN DATASET 
# df = pd.read_csv("data/raisin.csv")
# X = df.drop(columns=["label"])
# Y = df["label"]

# min_value = X.min()
# max_value = X.max()
# X = (X-min_value) / (max_value-min_value)

# X  = X.astype(np.float64)

#------------------------------------------------------------------------------

#TITANIC DATASET
df = pd.read_csv("data/titanic.csv")
X = df.drop(columns=["label"])
Y = df["label"]

# one-hote encode categorical data 
X = pd.get_dummies(X, columns=[
    "attr1_cat","attr2_cat"
])

# normalize numerical data
num_cols = ["attr3_num","attr4_num","attr5_num","attr6_num"]
min_val = X[num_cols].min()
max_val = X[num_cols].max()
X[num_cols] = (X[num_cols] - min_val) / (max_val - min_val)
X  = X.astype(np.float64)

#------------------------------------------------------------------------------

# #split data into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)



def strat_cross_validation(X,Y,k):
    #group data together
    label_map = {}
    for key in Y.unique():
        values = Y[Y == key].index.tolist()
        np.random.shuffle(values)
        label_map[key] = values

    #split the data between k folds 
    X_folds = []
    Y_folds = []
    for idx in range(k):
        X_folds.append([])
        Y_folds.append([])
    
    for key, cur_group in label_map.items():
        length = len(cur_group)
        num_in_fold = length // k  
        for i in range(k):
            start = i * num_in_fold
            end = (i+1) * num_in_fold if i < k-1 else length
            cur_fold = cur_group[start:end]

            for index in cur_fold:
                X_folds[i].append(X.loc[index])
                Y_folds[i].append(Y.loc[index])
        

    for i in range(len(X_folds)):
        X_folds[i] = pd.DataFrame(X_folds[i]).reset_index(drop=True)
        Y_folds[i] = pd.Series(Y_folds[i]).reset_index(drop=True)

    return X_folds, Y_folds

def initialize_weights(num_neurons_layer):
    weights = []
    for i in range(len(num_neurons_layer)-1):
        weights_layer = np.random.normal(0,1,size=(num_neurons_layer[i+1], num_neurons_layer[i]+1))
        weights.append(weights_layer)
    return weights

def train_neural_network(X,Y,layer_sizes,alpha,lam,num_iterations,batch_size = 30):
    weights = initialize_weights(layer_sizes) 
    
    for i in range(num_iterations):
        #shuffle data
        X, Y = shuffle(X, Y)

        #mini batch gradient descent
        for batch in range(0, len(X), batch_size):
            batch_end = batch + batch_size
            if batch_end > len(X):
                batch_end = len(X)
            X_batch = X[batch:batch_end]
            Y_batch = Y[batch:batch_end]

            gradients = compute_gradients(X_batch, weights, Y_batch, lam)
            weights = update_weights(gradients, weights, alpha)

    return weights

def predict_neural_net(X, weights):
    a_arr,_ = forward_prop(X, weights)
    output_layer = a_arr[-1]
    predictions = (output_layer >= 0.5).astype(int)
    return predictions

def calc_metrics(actual,predicted):
    TN = FN = FP = TP = 0
    for act, pred in zip(actual, predicted):
        if act == 0 and pred == 0:
            TN += 1
        elif act == 1 and pred == 0:
            FN += 1
        elif act == 0 and pred == 1:
            FP += 1
        else:
            TP += 1

    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return accuracy, precision, recall, f1_score

def evaluate_neural_network(X_data, Y_data, layer_sizes, alpha, lam, num_iterations, k=5):
    X_folds, Y_folds = strat_cross_validation(X_data, Y_data, k)

    fold_accuracies = []
    fold_precisions = []
    fold_recalls = []
    fold_f1s = []

    for round in range(k):
        X_train = []
        Y_train = []
        predictions = []

        for i in range(k):
            if i == round:
                X_test = X_folds[i]
                Y_test = Y_folds[i]
            else:
                X_train.append(X_folds[i])
                Y_train.append(Y_folds[i])

        X_train = pd.concat(X_train).reset_index(drop=True)
        Y_train = pd.concat(Y_train).reset_index(drop=True)

        X_train = X_train.to_numpy()
        Y_train = Y_train.to_numpy().reshape(-1, 1)

        X_test = X_test.to_numpy()
        Y_test = Y_test.to_numpy().reshape(-1, 1)

        weights = train_neural_network(X_train, Y_train, layer_sizes, alpha, lam, num_iterations)
        preds = predict_neural_net(X_test, weights)


        accuracy, precision, recall, f1_score = calc_metrics(Y_test, preds)

        fold_accuracies.append(accuracy)
        fold_precisions.append(precision)
        fold_recalls.append(recall)
        fold_f1s.append(f1_score)

    return (
        np.mean(fold_accuracies),
        np.mean(fold_precisions),
        np.mean(fold_recalls),
        np.mean(fold_f1s)
    )

#-------------------------------------------------------------------------------
#TESTING ARCHITECTURES

# input_size = X.shape[1]
# architectures = [
#     [input_size, 1, 1],
#     [input_size, 2, 1],
#     [input_size, 4, 1],
#     [input_size, 4, 4, 1],
#     [input_size, 8, 8, 1],
#     [input_size, 4, 4, 4, 1]
# ]

# lambda_values = [0.0, 0.1, 0.3]

# results = []

# for architecture in architectures:
#     for lamda in lambda_values:
#         accuracy, _, _, f1 = evaluate_neural_network(X, Y, layer_sizes=architecture, alpha=0.1, lam=lamda, num_iterations=500, k=5)
#         results.append({'Architecture': str(architecture),'Lambda': lamda,'Accuracy': accuracy,'F1 Score': f1})
# results_df = pd.DataFrame(results)
# print(results_df)

def learning_curve(X, Y, num_layers, lam, alpha, num_iterations):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
    X_train = X_train.to_numpy()
    Y_train = Y_train.to_numpy().reshape(-1, 1)
    X_test = X_test.to_numpy()
    Y_test = Y_test.to_numpy().reshape(-1, 1)

    training_samples = list(range(5, len(X_train) + 1, 20))

    J_arr = []

    for sample in training_samples:
        X_sample = X_train[:sample]
        Y_sample = Y_train[:sample]

        weights = train_neural_network(X_sample, Y_sample,layer_sizes=num_layers,alpha=alpha,lam=lam,num_iterations=num_iterations)

        a_arr, _ = forward_prop(X_test, weights)
        cost = compute_final_cost(a_arr[-1], Y_test, lam, weights)
        J_arr.append(cost)

    plt.figure(figsize=(8, 5))
    plt.plot(training_samples, J_arr, marker='o')
    plt.xlabel('Number of Training Examples')
    plt.ylabel('Cost (J) on Validation Set')
    plt.title(f'Learning Curve')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


learning_curve(X,Y,num_layers=[9,4,4,1],lam=0.0,alpha=0.1,num_iterations=500)









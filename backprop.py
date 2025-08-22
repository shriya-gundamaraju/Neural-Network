import numpy as np

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def forward_prop(data, weights):
    a_arr = []
    z_arr = []

    a = data

    for theta in weights:
        #add bias column 
        a_bias = np.insert(a, 0, 1, axis=1)
        #calculate the weighted sum
        z = np.dot(a_bias, theta.T)
        #activation function
        a = sigmoid(z)
        a_arr.append(a)
        z_arr.append(z)

    return a_arr, z_arr

def back_prop(data, weights, y):
    a_arr, z_arr = forward_prop(data, weights)

    #compute the delta values for all neurons in the output layer 
    delta_output = a_arr[-1] - y
    deltas = [delta_output]

    #compute the delta values for all neurons in the hidden layers
    for l in range(len(weights)-1,0,-1):
        #exclude bias column 
        theta = weights[l][:,1:]
        delta_hidden = np.dot(deltas[0], theta) * a_arr[l-1] * (1 - a_arr[l-1])
        deltas.insert(0, delta_hidden)

    return deltas 

def compute_gradients(data,weights,y,lam):
    #number of training examples
    n = data.shape[0]
    #list of activations for each layer 
    a_arr, z_arr = forward_prop(data, weights)
    #list of delta values for each layer
    deltas = back_prop(data, weights, y)
    gradients = []
    for l in range(len(weights)):
        #add bias column to activations 
        if l == 0:
            # input layer
            activation_bias = data
            activation_bias = np.insert(activation_bias, 0, 1, axis=1)
        else:
            activation_bias = a_arr[l-1]
            activation_bias = np.insert(activation_bias, 0, 1, axis=1)

        #error term for layer l+1
        delta = deltas[l]
        #compute the gradient for layer
        gradient = np.dot(delta.T, activation_bias) 
        #add regularization term to gradient
        reg = lam * weights[l]  
        #do not regularize the bias term
        reg[:, 0] = 0   
        #combines gradients with regularization term and divides by #instances to obtain average gradient                 
        gradient = (gradient + reg) / n  
        gradients.append(gradient)
    
    return gradients

def update_weights(gradients,weights,alpha):
    #updates the weights of each layer based on their corresponding gradients 
    for i in range(len(weights)):
        weights[i] = weights[i] - (alpha * gradients[i])
    return weights

def compute_cost(pred, true):
    pred = np.array(pred)
    return float(np.sum(-true * np.log(pred) - (1 - true) * np.log(1 - pred)))


def compute_final_cost(preds, y, lam, weights):
    n = len(y)
    #didves the total error/cost of the network by the number of training instances
    J = compute_cost(preds, y) / n
    #computes the square of all the weights of the network (except bias wieghts) and adds them up
    S = 0
    for theta in weights:
        S += np.sum(np.square(theta[:, 1:]))
    #computes the term used to regularize the network's cost
    S = (lam / (2 * n)) * S
    return J + S


def backprop_1():
    x = np.array([[0.13], [0.42]])
    y = np.array([[0.9], [0.23]])
    lambda_val = 0.0

    weights = [np.array([[0.4, 0.1],
                         [0.3, 0.2]]),
               np.array([[0.7, 0.5, 0.6]])]

    a_arr, z_arr = forward_prop(x, weights)
    #predictions based on forward propagation
    preds = a_arr[-1]
    print("--------------------------------------------")
    print("\n Computing the error/cost, J, of the network")
    for i in range(len(x)):
        print("Processing training instance ", i+1)
        print("Forward propagating the input", x[i])
        pred = preds[i].item()
        actual = y[i].item()
        cost = compute_cost(pred, actual)
        final_cost = compute_final_cost(preds, y, lambda_val, weights)
        print(f"  a1: [1.00000, {x[i][0]:.5f}]")
        print(f"  z2: {np.array2string(z_arr[0][i], precision=5, floatmode='fixed')}")
        print(f"  a2: [1.00000, {a_arr[0][i][0]:.5f}, {a_arr[0][i][1]:.5f}]")
        print(f"  z3: {np.array2string(z_arr[1][i], precision=5, floatmode='fixed')}")
        print(f"  a3: {float(pred):.5f}")
        print(f"  f(x): {float(pred):.5f}")
        print(f"Predicted output for instance {i+1}: {float(pred):.5f}")
        print(f"Expected output for instance {i+1}: {float(actual):.5f}")
        print(f"Cost, J associated with instance {i+1}: {(cost):.3f}")
        print()
    print(f"Final (regularized) cost, J, based on the complete training set: {final_cost:.5f}")
    print("--------------------------------------------")
    # Backward propagation
    deltas = back_prop(x, weights, y)
    gradients = compute_gradients(x, weights, y, lambda_val)
    print("Running backpropagation")
    for i in range(len(x)):
        print(f"Computing gradients based on training instance {i+1}")
        print(f"delta3: {np.round(deltas[-1][i], 5).tolist()}")
        print(f"delta2: {np.round(deltas[0][i], 5).tolist()}")
        print()
        print(f"Gradients of Theta2 based on training instance {i+1}:")
        gradient_theta2 = np.outer(deltas[-1][i], np.hstack(([1], a_arr[0][i])))
        print(f"{np.round(gradient_theta2, 5).tolist()}")
        print()
        print(f"Gradients of Theta1 based on training instance {i+1}:")
        gradient_theta1 = np.outer(deltas[0][i], np.hstack(([1], x[i])))
        print(f"{np.round(gradient_theta1, 5).tolist()}")
        print()

    print("The entire training set has been processed. Computing the average (regularized) gradients:")
    avg_gradients = []
    for grad in gradients:
        grad = np.round(grad, 5)
        avg_gradients.append(grad)
    print(f"Final regularized gradients of Theta1:")
    print(f"{avg_gradients[0].tolist()}")
    print()
    print(f"Final regularized gradients of Theta2:")
    print(f"{avg_gradients[1][0].tolist()}")

backprop_1()

def backprop_2():
    x = np.array([[0.32, 0.68],
                  [0.83, 0.02]])
    y = np.array([[0.75, 0.98],
                  [0.75, 0.28]])
    lambda_val = 0.25

    weights = [
        np.array([[0.42, 0.15, 0.40],
                  [0.72, 0.10, 0.54],
                  [0.01, 0.19, 0.42],
                  [0.30, 0.35, 0.68]]),
        np.array([[0.21, 0.67, 0.14, 0.96, 0.87],
                  [0.87, 0.42, 0.20, 0.32, 0.89],
                  [0.03, 0.56, 0.80, 0.69, 0.09]]),
        np.array([[0.04, 0.87, 0.42, 0.53],
                  [0.17, 0.10, 0.95, 0.69]])
    ]

    a_arr, z_arr = forward_prop(x, weights)
    preds = a_arr[-1]
    print("--------------------------------------------")
    print("\nComputing the error/cost, J, of the network")
    for i in range(2):
        print("Processing training instance ", i+1)
        print("Forward propagating the input", np.round(x[i],5).tolist())
        pred = preds[i]
        actual = y[i]
        cost = compute_cost(pred, actual)
        final_cost = compute_final_cost(preds, y, lambda_val, weights)

        print(f"  a1: [1.00000, {x[i][0]:.5f}, {x[i][1]:.5f}]")
        print(f"  z2: {np.round(z_arr[0][i],5).tolist()}")
        print(f"  a2: [1.00000, {np.round(a_arr[0][i],5).tolist()}]")
        print(f"  z3: {np.round(z_arr[1][i],5).tolist()}")
        print(f"  a3: [1.00000, {np.round(a_arr[1][i],5).tolist()}]")
        print(f"  z4: {np.round(z_arr[2][i],5).tolist()}")
        print(f"  a4: {np.round(a_arr[2][i],5).tolist()}")
        print(f"f(x): {np.round(pred,5).tolist()}")
        print(f"Predicted output for instance {i+1}: {np.round(pred,5).tolist()}")
        print(f"Expected output for instance {i+1}: {np.round(actual,5).tolist()}")
        print(f"Cost, J associated with instance {i+1}: {np.round(cost,3).tolist()}")
        print()
    
    print(f"Final (regularized) cost, J, based on the complete training set: {np.round(final_cost,5).tolist()}")
    # Backward Propagation
    deltas = back_prop(x, weights, y)
    gradients = compute_gradients(x, weights, y, lambda_val)
    print("--------------------------------------------")
    print("\nRunning backpropagation")
    for i in range(len(x)):
        print(f"Computing gradients based on training instance {i+1}")
        print(f"delta4: {np.round(deltas[-1][i], 5).tolist()}")
        print(f"delta3: {np.round(deltas[-2][i], 5).tolist()}")
        print(f"delta2: {np.round(deltas[-3][i], 5).tolist()}")
        print()


        print(f"Gradients of Theta3 based on training instance {i+1}:")
        gradient_theta3 = np.outer(deltas[-1][i], np.hstack(([1], a_arr[-2][i])))
        print(np.round(gradient_theta3,5).tolist())
        print()

        print(f"Gradients of Theta2 based on training instance {i+1}:")
        gradient_theta2 = np.outer(deltas[-2][i], np.hstack(([1], a_arr[-3][i])))
        print(np.round(gradient_theta2,5).tolist())
        print()

        print(f"Gradients of Theta1 based on training instance {i+1}:")
        gradient_theta1 = np.outer(deltas[-3][i], np.hstack(([1], x[i])))
        print(np.round(gradient_theta1,5).tolist())
        print()

    print("The entire training set has been processed. Computing the average (regularized) gradients:")
    avg_gradients = []
    for grad in gradients:
        grad = np.round(grad, 5)
        avg_gradients.append(grad)

    print(f"Final regularized gradients of Theta1:")
    print(avg_gradients[0].tolist())
    print()

    print(f"Final regularized gradients of Theta2:")
    print(avg_gradients[1].tolist())
    print()

    print(f"Final regularized gradients of Theta3:")
    print(avg_gradients[2].tolist())

backprop_2()







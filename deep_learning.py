import numpy as np
import pandas as pd


def sigmoid(Z):
    A = 1 / (1 + np.exp(-Z))
    cache = Z

    return A, cache


def sigmoid_backward(dA, cache):
    Z = cache

    s = 1 / (1 + np.exp(-Z))
    dZ = dA * s * (1 - s)

    assert (dZ.shape == Z.shape)

    return dZ


def relu(Z):
    A = np.maximum(0, Z)

    assert (A.shape == Z.shape)

    cache = Z
    return A, cache


def relu_backward(dA, cache):
    Z = cache
    dZ = np.array(dA, copy=True)  # just converting dz to a correct object.

    # When z <= 0, you should set dz to 0 as well.
    dZ[Z <= 0] = 0

    assert (dZ.shape == Z.shape)

    return dZ


def inicializa_parametros_deep(layer_dims):
    np.random.seed(1)
    parameters = {}
    L = len(layer_dims)  # number of layers in the network

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) / np.sqrt(
            layer_dims[l - 1])  # *0.01
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1)) + .01

        assert (parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l - 1]))
        assert (parameters['b' + str(l)].shape == (layer_dims[l], 1))

    return parameters


def Tamanho_Camadas(X):
    n_x = X.shape[1]
    n_h = 2
    n_y = X.shape[1]

    return (n_x, n_h, n_y)


def para_frente_linear(A, W, b):
    Z = np.dot(W, A) + b

    assert (Z.shape == (W.shape[0], A.shape[1]))
    cache = (A, W, b)

    return Z, cache


def para_frente_linear_ativacao(A_prev, W, b, activation):
    if activation == "sigmoid":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        Z, linear_cache = para_frente_linear(A_prev, W, b)
        A, activation_cache = sigmoid(Z)

    elif activation == "relu":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        Z, linear_cache = para_frente_linear(A_prev, W, b)
        A, activation_cache = relu(Z)

    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)

    return A, cache


def modelo_para_frente_L(X, parameters):
    caches = []
    A = X
    L = len(parameters) // 2  # number of layers in the neural network

    # Implement [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.
    for l in range(1, L):
        A_prev = A
        A, cache = para_frente_linear_ativacao(A_prev, parameters['W' + str(l)], parameters['b' + str(l)],
                                               activation="sigmoid")
        caches.append(cache)

    # Implement LINEAR -> SIGMOID. Add "cache" to the "caches" list.
    AL, cache = para_frente_linear_ativacao(A, parameters['W' + str(L)], parameters['b' + str(L)], activation="sigmoid")
    caches.append(cache)

    return AL, caches


def compute_custo(AL, Y):
    m = Y.shape[1]

    cost = (1. / m) * (-np.dot(Y, np.log(AL).T) - np.dot(1 - Y, np.log(1 - AL).T))

    cost = np.squeeze(cost)  # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).

    return cost


def para_tras_linear(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = (1 / m) * np.dot(dZ, A_prev.T)
    db = 1. / m * np.sum(dZ, axis=1, keepdims=True)
    dA = np.dot(W.T, dZ)

    assert (dA.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)

    return dA, dW, db


def para_tras_linear_ativacao(dA, cache, activation):
    linear_cache, activation_cache = cache

    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = para_tras_linear(dZ, linear_cache)

    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = para_tras_linear(dZ, linear_cache)

    return dA_prev, dW, db


def modelo_para_tras_L(AL, Y, caches):
    grads = {}
    L = len(caches)  # the number of layers
    m = AL.shape[1]

    # Initializing the backpropagation
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

    # Lth layer (SIGMOID -> LINEAR) gradients. Inputs: "AL, Y, caches". Outputs: "grads["dAL"], grads["dWL"], grads["dbL"]
    current_cache = caches[L - 1]
    grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = para_tras_linear_ativacao(dAL, current_cache,
                                                                                                 activation="sigmoid")

    for l in reversed(range(L - 1)):
        # lth layer: (RELU -> LINEAR) gradients.
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = para_tras_linear_ativacao(grads["dA" + str(l + 2)], current_cache,
                                                                   activation="sigmoid")
        grads["dA" + str(l + 1)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads


def atualizacao_parametros(parameters, grads, learning_rate):
    L = len(parameters) // 2  # number of layers in the neural network

    for l in range(L):
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]

    return parameters


def modelo_L_camadas(X, Y, layers_dims, learning_rate=0.0075, num_iterations=3000, print_cost=False):  # lr was 0.009

    np.random.seed(1)
    costs = []

    parameters = inicializa_parametros_deep(layers_dims)

    for i in range(0, num_iterations):
        AL, caches = modelo_para_frente_L(X, parameters)

        cost = compute_custo(AL, Y)

        grads = modelo_para_tras_L(AL, Y, caches)

        atualizacao_parametros(parameters, grads, learning_rate)

        if print_cost and i % 100 == 0:
            print("Cost after iteration ", cost)
        if print_cost and i % 100 == 0:
            costs.append(cost)

    print("Taxa de aprendizado =" + str(learning_rate))

    return parameters


def prever(X, y, parameters):

    m = X.shape[1]
    n = len(parameters) // 2  # number of layers in the neural network
    p = np.zeros((1, m))

    # Forward propagation
    probas, caches = modelo_para_frente_L(X, parameters)

    # convert probas to 0/1 predictions
    for i in range(0, probas.shape[1]):
        if probas[0, i] > 0.5:
            p[0, i] = 1
        else:
            p[0, i] = 0

    # print results
    # print ("predictions: " + str(p))
    # print ("true labels: " + str(y))
    print("Accuracy: " + str(np.sum((p == y) / m)))

    return p


dataset = pd.read_csv('dataset.csv', low_memory=False,
                      usecols=['year', 'vote_count', 'vote_average', 'popularity', 'Science Fiction', 'Action', 'War',
                               'Adventure', 'Horror', 'Thriller', 'History', 'Documentary', 'Romance',
                               'Music', 'Drama', 'Animation', 'Mystery', 'Western', 'TV Movie', 'Fantasy', 'Comedy',
                               'Crime', 'Family'])
X_assess = dataset
Y_assess = np.ones((1, X_assess.shape[0]))
layers_dims = [X_assess.shape[1], X_assess.shape[1], X_assess.shape[1]]
parameters = modelo_L_camadas(X_assess.T, Y_assess, layers_dims=layers_dims, num_iterations=5000, print_cost=True)

# prob, caches = modelo_para_frente_L(X_assess, parameters)
p = prever(X_assess[:1].T, Y_assess, parameters)
# print(prob)
print(p)
# print("W1 = " + str(parameters["W1"]))
# print("b1 = " + str(parameters["b1"]))
# print("W2 = " + str(parameters["W2"]))
# print("b2 = " + str(parameters["b2"]))

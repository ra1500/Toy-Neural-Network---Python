# Toy Neural Network - Python only - 1 hidden layer
# Follow step-by-step the setting up of a neural network from scratch. Then optimize/train it through a for loop.

import numpy as np

# import matplotlib.pyplot as plt

# ---- create some (nonlinear) data to train a nn classifier/predictor ----

N = 100  # number of points per class
D = 2  # dimensionality/attributes/pixels
K = 3  # number of classes
X = np.zeros((N * K, D))  # data matrix (each row = single example)
y = np.zeros(N * K, dtype='uint8')  # class labels
for j in range(K):
    ix = range(N * j, N * (j + 1))
    r = np.linspace(0.0, 1, N)  # radius
    t = np.linspace(j * 4, (j + 1) * 4, N) + np.random.randn(N) * 0.2  # theta
    X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
    y[ix] = j

# visualize the data:
# plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
# plt.show()

# ---- data preprocessing ----
# centering and normalizing. unnecessary here since this data is already formatted

# ---- setup the parameters (weights and bias) ----
h = 100  # size of hidden layer
W = 0.01 * np.random.randn(D, h)
b = np.zeros((1, h))
W2 = 0.01 * np.random.randn(h, K)
b2 = np.zeros((1, K))

print()
print('X matrix, N x D, first 3 rows/samples, ', X.shape)
print(X[:3])
print()
print('W matrix, D x h, ', W.shape)
print('W2 matrix, h x k, ', W2.shape)
print('Bias vector, 1 x h ', b.shape)
print('Bias2 vector, 1 x k ', b2.shape)

# ---- Scores/Forward Pass ----
hidden_layer = np.maximum(0, np.dot(X, W) + b)  # ReLU activation
scores = np.dot(hidden_layer, W2) + b2

print()
print('Scores, first 3 rows,  ', scores.shape)
print(scores[:3])

# ---- Losses (softmax classifier) with L2 regularization
exp_scores = np.exp(scores) # get unnormalized probabilities
probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) # normalize them for each example so that total probability = 1
print()
print('Probabilities of scores', probs.shape)
print(probs[:3])

# probabilities of correct class
num_examples = X.shape[0]
correct_logprobs = -np.log(probs[range(num_examples), y])
print()
print('Probabilities of correct class', correct_logprobs.shape)
print(correct_logprobs[:10])

# average cross-entropy loss
data_loss = np.sum(correct_logprobs) / num_examples
print()
print('Data Loss before regularization')
print(data_loss)

# Regularization - L2
reg = 1e-3
reg_loss = 0.5 * reg * np.sum(W * W)

# Loss (data + regularization)
loss = data_loss + reg_loss
print()
print('Loss')
print(loss)

# ---- Gradiant Calculation ----
dscores = probs
dscores[range(num_examples), y] -= 1
dscores /= num_examples
dhidden = np.dot(dscores, W2.T)
print()
print('Gradiant matrix of W2, sample rows x class columns', dscores.shape)
print(dscores[:3])
print()
print('Gradiant matrix of W, sample rows x class columns', dhidden.shape)
print(dhidden[0, 0])

# ---- Backpropagation (partial derivative of the score with respect to the weights)----
dW2 = np.dot(hidden_layer.T, dscores)
db2 = np.sum(dscores, axis=0, keepdims=True)
dhidden = np.dot(dscores, W2.T)
dhidden[hidden_layer <= 0] = 0  # backprop the ReLU non-linearity
dW = np.dot(X.T, dhidden)
db = np.sum(dhidden, axis=0, keepdims=True)
dW2 += reg * W2
dW += reg * W  # the regularization gradient
print()

# ---- Parameter (weights) update ----
step_size = 1e-0
W += dW * -step_size
b += db * -step_size
W2 += dW2 * -step_size
b2 += db2 * -step_size
print('Step size/Learning Rate ', step_size)
print('Regularization ', reg)
print()

# ********* Above was the first training pass. To complete training via a for loop, this is continued/repeated in the below ********

for i in range(3001):
    hidden_layer = np.maximum(0, np.dot(X, W) + b)
    scores = np.dot(hidden_layer, W2) + b2
    exp_scores = np.exp(scores)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    correct_logprobs = -np.log(probs[range(num_examples), y])
    data_loss = np.sum(correct_logprobs) / num_examples
    reg_loss = 0.5 * reg * np.sum(W * W) + 0.5 * reg * np.sum(W2 * W2)
    loss = data_loss + reg_loss
    if i % 500 == 0:
        print("iteration %d: loss %f" % (i, loss))
    dscores = probs
    dscores[range(num_examples), y] -= 1
    dscores /= num_examples
    dW2 = np.dot(hidden_layer.T, dscores)
    db2 = np.sum(dscores, axis=0, keepdims=True)
    dhidden = np.dot(dscores, W2.T)
    dhidden[hidden_layer <= 0] = 0
    dW = np.dot(X.T, dhidden)
    db = np.sum(dhidden, axis=0, keepdims=True)
    dW2 += reg * W2
    dW += reg * W
    W += -step_size * dW
    b += -step_size * db
    W2 += -step_size * dW2
    b2 += -step_size * db2

# ---- Accuracy ----
predicted_class = np.argmax(scores, axis=1)
print()
print('Training accuracy: %.2f percent' % (np.mean(predicted_class == y) * 100))

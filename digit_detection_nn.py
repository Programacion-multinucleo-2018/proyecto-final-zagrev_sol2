import numpy as np
from scipy import optimize
import random
import time

class Neural_Network(object):
    def __init__(self, input, hidden, output):
        self.input_layer_size = input;
        self.hidden_layer_size = hidden;
        self.output_layer_size = output;

        self.W1 = np.random.randn(self.input_layer_size,self.hidden_layer_size)
        self.W2 = np.random.randn(self.hidden_layer_size,self.output_layer_size)

    def forward(self, X):
        self.z2 = np.dot(X, self.W1)
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.W2)
        yHat = self.sigmoid(self.z3)
        return yHat

    def sigmoid(self, z):
        return 1/(1+np.exp(-z))

    def sigmoidPrime(self,z):
        return np.exp(-z)/((1+np.exp(-z))**2)

    def costFunction(self, X, y):
        self.yHat = self.forward(X)
        J = 0.5*sum((y-self.yHat)**2)
        return J

    def costFunctionPrime(self, X, y):
        self.yHat = self.forward(X)

        delta3 = np.multiply(-(y-self.yHat), self.sigmoidPrime(self.z3))
        dJdW2 = np.dot(self.a2.T, delta3)

        delta2 = np.dot(delta3, self.W2.T)*self.sigmoidPrime(self.z2)
        dJdW1 = np.dot(X.T, delta2)

        return dJdW1, dJdW2

    def getParams(self):
        params = np.concatenate((self.W1.ravel(), self.W2.ravel()))
        return params

    def setParams(self, params):
        W1_start = 0
        W1_end = self.hidden_layer_size * self.input_layer_size
        self.W1 = np.reshape(params[W1_start:W1_end], (self.input_layer_size , self.hidden_layer_size))
        W2_end = W1_end + self.hidden_layer_size*self.output_layer_size
        self.W2 = np.reshape(params[W1_end:W2_end], (self.hidden_layer_size, self.output_layer_size))

    def computeGradients(self, X, y):
        dJdW1, dJdW2 = self.costFunctionPrime(X, y)
        return np.concatenate((dJdW1.ravel(), dJdW2.ravel()))

class Trainer(object):
    def __init__(self, N):
        self.N = N

    def callbackF(self, params):
        self.N.setParams(params)
        self.J.append(self.N.costFunction(self.X, self.y))

    def costFunctionWrapper(self, params, X, y):
        self.N.setParams(params)
        cost = self.N.costFunction(X, y)
        grad = self.N.computeGradients(X,y)

        return cost, grad

    def train(self, X, y):
        self.X = X
        self.y = y

        self.J = []

        params0 = self.N.getParams()

        options = {'maxiter': 200, 'disp' : True}
        _res = optimize.minimize(self.costFunctionWrapper, params0, jac=True, method='BFGS', args=(X, y), options=options, callback=self.callbackF)

        self.N.setParams(_res.x)
        self.optimizationResults = _res

training_set = []
results = []

file = open('digitos.txt', 'r')

for line in file:
    digits = line.split(' ')
    training_set.append(np.array([float(i) for i in digits[:400]]))
    results.append([float(digits[400])])

X = np.asarray(training_set)
y = np.asarray(results)

y = y/10

indices = np.arange(X.shape[0])
np.random.shuffle(indices)

X = X[indices]
y = y[indices]

start_time = time.time()
NN = Neural_Network(400, 25, 1)

T = Trainer(NN)
T.train(X[:4000],y[:4000])
elapsed_time = time.time() - start_time

i = 4000
for x in X[-1000:]:
    result = NN.forward(x)
    print(result, y[i])
    i += 1

print('Tiempo: ' + str(elapsed_time) + ' s')

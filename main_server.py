import pandas as pd
import time
import matplotlib.pyplot as plt
import numpy as np
from keras.utils import np_utils
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from keras import backend as K
import NeuraNetwork
from PIL import Image
import pickle
import socket

def tanh(x):
  return np.tanh(x)


def tanh_prime(x):
  return 1 - np.tanh(x) ** 2


def relu_deriv(x):
  x[x <= 0] = 0
  x[x > 0] = 1
  return x


def mse(y_true, y_pred):
  return np.mean(np.power(y_true - y_pred, 2))


def mse_prime(y_true, y_pred):
  return 2 * (y_pred - y_true) / y_true.size


def recall_m(y_true, y_pred):
  true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
  possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
  recall = true_positives / (possible_positives + K.epsilon())
  return recall


def precision_m(y_true, y_pred):
  true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
  predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
  precision = true_positives / (predicted_positives + K.epsilon())
  return precision


def f1_m(y_true, y_pred):
  precision = precision_m(y_true, y_pred)
  recall = recall_m(y_true, y_pred)
  return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


path = "/Users/aly/Desktop/ahcd1/"

'''

my_data = pd.read_csv("/Users/Aly/Desktop/ahcd1_orig/csvTrainImages.csv" , header=None)
labels_train = pd.read_csv("/Users/Aly/Desktop/ahcd1_orig/csvTrainLabel.csv")
testing_data = pd.read_csv("/Users/Aly/Desktop/ahcd1_orig/csvTestImages.csv" , header=None)
test_labels = pd.read_csv("/Users/Aly/Desktop/ahcd1_orig/csvTestLabel.csv")

'''

my_data = pd.read_csv(path + "training.csv")
labels_train = pd.read_csv(path + "train_labels.csv")
testing_data = pd.read_csv(path + "testing.csv")
test_labels = pd.read_csv(path + "test_labels.csv")

all_data = pd.concat([my_data, testing_data], axis=0, sort=False)
all_labels = pd.concat([labels_train, test_labels], axis=0, sort=False)

scaler = MinMaxScaler()
# Fit on training set only.
scaler.fit(my_data)
# Apply transform to both the training set and the test set.
my_data = scaler.transform(my_data)
testing_data = scaler.transform(testing_data)
# pca = PCA(0.95)
# pca.fit(my_data)
# my_data = pca.transform(my_data)
# testing_data = pca.transform(testing_data)

'''
y_train = np.array(labels_train)
y_test = np.array(test_labels)
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
y_train = y_train[:,1:]
y_test = y_test[:,1:]
'''

X_train = np.array(my_data)
X_test = np.array(testing_data)
X_train = X_train.reshape(X_train.shape[0], -1).T
X_test = X_test.reshape(X_test.shape[0], -1).T
X_test = X_test.astype('float32')
X_train = X_train.astype('float32')
y_train = np.array(labels_train)
y_test = np.array(test_labels)

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
y_train = y_train[:, 1:]
y_test = y_test[:, 1:]
y_test = y_test.T
y_train = y_train.T


#test_labels = np.array(test_labels['labels'])
#labels_train = np.array(labels_train['labels'])


plt.rcParams['figure.figsize'] = (5.0, 4.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

np.random.seed(1)

layers_dims = [1024, 256, 256, 28]
keep_prob = [1, 0.7, 0.7, 1]
m = X_train.shape[0]


def L_layer_model(X, Y, layers_dims, learning_rate=0.13, num_iterations=2500, print_cost=False,
                  hidden_layers_activation_fn="relu"):  # lr was 0.009

  m = X.shape[1]
  np.random.seed(1)
  costs = []  # keep track of cost

  parameters = NeuraNetwork.initialize_parameters_deep(layers_dims)

  for i in range(0, num_iterations):

    D = NeuraNetwork.drop_out_matrices(layers_dims, m, keep_prob)

    # compute forward propagation
    AL, caches = NeuraNetwork.L_model_forward(
      X, parameters, D, keep_prob, hidden_layers_activation_fn)

    # compute regularized cost
    cost = NeuraNetwork.compute_cost(AL, Y)

    # compute gradients
    grads = NeuraNetwork.L_model_backward(
      AL, Y, caches, D, keep_prob, hidden_layers_activation_fn)

    parameters = NeuraNetwork.update_parameters(parameters, grads, learning_rate)

    if print_cost and i % 100 == 0:
      print("Cost after iteration %i: %f" % (i, cost))
    if print_cost and i % 100 == 0:
      costs.append(cost)

  # plot the cost
  plt.plot(np.squeeze(costs))
  plt.ylabel('cost')
  plt.xlabel('iterations (per tens)')
  plt.title("Learning rate =" + str(learning_rate))
  plt.show()

  return parameters


def take_inp(filename):
  img_file = Image.open(filename)
  width, height = img_file.size
  value = np.array(img_file, dtype=np.uint8).reshape((width, height))

  value = value.flatten(order='C')
  value = value / 255
  value = value.reshape(value.shape[0], 1)

  return value

def print_letter(number):
  letter = ''
  if number == 0:
    letter = 'alef'

  if number == 1:
    letter = 'ba'

  if number == 2:
    letter = 'ta'

  if number == 3:
    letter = 'tha'

  if number == 4:
    letter = 'gim'

  if number == 5:
    letter = 'ha'

  if number == 6:
    letter = 'kha'

  if number == 7:
    letter = 'dal'

  if number == 8:
    letter = 'zal'

  if number == 9:
    letter = 'ra'

  if number == 10:
    letter = 'zain'

  if number == 11:
    letter = 'sin'

  if number == 12:
    letter = 'shin'

  if number == 13:
    letter = 'sad'

  if number == 14:
    letter = 'dad'

  if number == 15:
    letter = 'tah'

  if number == 16:
    letter = 'zah'

  if number == 17:
    letter = 'ayn'

  if number == 18:
    letter = 'ghayn'

  if number == 19:
    letter = 'fa'

  if number == 20:
    letter = 'qaf'

  if number == 21:
    letter = 'kaf'

  if number == 22:
    letter = 'lam'

  if number == 23:
    letter = 'mim'

  if number == 24:
    letter = 'nun'

  if number == 25:
    letter = 'ha'

  if number == 26:
    letter = 'waw'

  if number == 27:
    letter = 'yeh'

  return letter


'''
parameters = L_layer_model(X_train, y_train, layers_dims, num_iterations=1500, print_cost=True)
f = open('store.pckl', 'wb')
pickle.dump(parameters, f)
f.close()




f = open('store.pckl', 'rb')
parameters = pickle.load(f)
f.close()

NeuraNetwork.predict(X_train, y_train,parameters)
NeuraNetwork.predict(X_test, y_test,parameters)




'''




port = 60000  # Reserve a port for your service.
s = socket.socket()  # Create a socket object
host = socket.gethostname()  # Get local machine name
s.bind((host, port))  # Bind to the port
s.listen(5)  # Now wait for client connection.

print('Server listening....')

while True:
  conn, addr = s.accept()  # Establish connection with client.
  print('Got connection from', addr)
  data = conn.recv(1024)
  print('Server received', repr(data))

  img = conn.recv(1024)
  print('recieved Image file name')
  img = img.decode("utf-8")
  test_input = take_inp(img)
  f = open('store.pckl', 'rb')
  parameters = pickle.load(f)
  f.close()
  result = NeuraNetwork.predict_one(test_input, parameters)

  out = print_letter(result)
  conn.sendall(out.encode('utf-8'))
  conn.close()


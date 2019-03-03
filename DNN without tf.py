import numpy as np
import os
from PIL import Image
os.chdir("C:/Users/엄현호/PycharmProjects/")
from utils import *
from utils2 import *
from sklearn.model_selection import train_test_split
import random

###################################################
###  폴더에 있는 알파벳 이미지를 행렬로 바꾸는 함수 ###
###################################################
def image_to_array(num_image, num_px, num_alphabets):

    alphabet_array = ['A', 'B', 'C','D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
                      'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']



    # 1. 행렬 초기화(뼈대 만들기)
    images_array = np.zeros((num_image*num_alphabets, num_px*num_px*3))

    # 2. 이미지의 pixel을 변환한 후, 행렬에 이미지 pixel 값 넣기
    for i in range(0, num_alphabets):
        os.chdir("C:/Users/엄현호/Desktop/새 폴더/asl-alphabet/asl_alphabet_train/"
                 + str(alphabet_array[i]))
        for j in range(0, num_image):
            # change pixels
            image = Image.open(alphabet_array[i] + str(j + 1) + ".jpg")
            image = image.resize((num_px, num_px), Image.ANTIALIAS)
            # save to matrix
            image = np.array(image)
            image_flatten = image.reshape(num_px*num_px*3)
            images_array[j + num_image*i] = image_flatten

    # 3. 행 열 뒤집기, 각 column이 이미지 1개를 나타내도록 해야 모델에 돌릴수 있음!
    images_array = images_array.T / 255

    return images_array


# y을 만드는 code
def make_answer(array, num_alphabets = 2): # array : np.array
    """
    ex) A, B 두 가지 image 종류일 때 : A -> 0, B -> 1의 값을 부여하는 np.array를 만들자!
    0~3000 : 0, 3001 ~ 6000 : 1 인 array 이런 식으로...
    """
    k = array.shape[1]
    answer_array = np.zeros((k, 1))
    for i in range(int(k/2), k):
        answer_array[i] = 1
    return answer_array

def L_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=False):#lr was 0.009
    """
    Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.

    Arguments:
    X -- data, numpy array of shape (number of examples, num_px * num_px * 3)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
    learning_rate -- learning rate of the gradient descent update rule
    num_iterations -- number of iterations of the optimization loop
    print_cost -- if True, it prints the cost every 100 steps

    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """

    np.random.seed(1)
    costs = []                         # keep track of cost

    # Parameters initialization. (≈ 1 line of code)
    parameters = initialize_parameters_deep(layers_dims)
    # Loop (gradient descent)
    for i in range(0, num_iterations):

        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
        AL, caches = L_model_forward(X, parameters)
        # Compute cost.

        cost = compute_cost(AL, Y)

        # Backward propagation
        grads = L_model_backward(AL, Y, caches)
        # Update parameters.

        parameters = update_parameters(parameters, grads, learning_rate)

        # Print the cost every 100 training example
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)

    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    return parameters


##### model에 돌리기 위한 전처리 코드 #####
## Constants Defining the model
# make full X
X = image_to_array(num_image= 3000, num_px=64, num_alphabets=2)
# make full y (A : 0, B : 1)
y = make_answer(X)

# make X, y train, test data sets
X_train, X_test, y_train, y_test = train_test_split(X.T,y, test_size=0.33)
X_train = X_train.T
X_test = X_test.T
# check matrix's dimensions are right
y_train = y_train.T
y_test = y_test.T
print("X_train : " + str(X_train.shape))
print("X_test : " + str(X_test.shape))
print("y_train : " + str(y_train.shape))
print("y_test : " + str(y_test.shape))
#X_train, X_test = train_test_split(X, y, test_size= 0.3, random_state= 42)

## define the number of input, hidden, output layers
layers_dims = [12288, 7, 5, 3, 1]

##### L-layer 신경망 model에 data 집어넣기 #####
"""
overfitting 막는 법 아무것도 안집어넣은 model
"""
parameters = L_layer_model(X_train,y_train, layers_dims,
                           num_iterations = 2000, print_cost = True)
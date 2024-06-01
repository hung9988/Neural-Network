import numpy as np
from keras.utils import to_categorical
from keras.datasets import mnist # type: ignore
from scipy.signal import correlate2d, convolve2d
class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, input):
        # TODO: return output
        pass

    def backward(self, output_gradient, learning_rate):
        # TODO: update parameters and return input gradient
        pass

class Activation(Layer):
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime

    def forward(self, input):
        self.input = input
        return self.activation(self.input)

    def backward(self, output_gradient, learning_rate):
        return np.multiply(output_gradient, self.activation_prime(self.input))


class Conv():
    def __init__(self, input_shape, kernel_size, num_kernels):
        self.input_shape = input_shape
        input_depth, input_height, input_width = input_shape
        self.kernel_size = kernel_size
        self.num_kernels = num_kernels
        self.input_depth= input_depth
        self.kernels = np.random.randn(num_kernels,input_depth, kernel_size, kernel_size)
        self.output_shape = (num_kernels, input_depth, input_height-kernel_size+1, input_width-kernel_size+1)
        self.biases= np.random.rand(*self.output_shape)
        
    def forward(self, input):
        self.input= input
    
        self.output= np.copy(self.biases)
        for i in range(self.num_kernels):
            for j in range(self.input_depth):
                self.output[i, j] += correlate2d(input[j], self.kernels[i,j], 'valid')
 
        return self.output
    def backward(self, grad, lr):
        dL_dW=np.zeros(self.kernels.shape)
        dL_dA=np.zeros(self.input.shape)
    
        for i in range(self.num_kernels):
            for j in range(self.input_depth):
                
                
                
                dL_dW[i,j] =correlate2d(self.input[j], grad[i,j], 'valid')
                dL_dA[j]+= convolve2d(grad[i,j], self.kernels[i,j], 'full')
                
        self.kernels -= lr*dL_dW
        self.biases -= lr*grad
        return dL_dA
    
class Dense(Layer):
    def __init__(self, input_shape, output_shape,alpha):
        self.alpha=alpha        
        self.w = np.random.randn(input_shape, output_shape)
        self.b=np.random.rand(1,output_shape)
        
    def forward(self,input):
        self.input=input.reshape(1,-1)
        result=np.matmul(self.input,self.w) + self.b
      
        return result
    
    def backward(self,grad,lr):
        
        dL_dw=(np.matmul(self.input.T,grad) + self.alpha*self.w)/self.input.shape[0]
        dL_db=np.mean(grad,axis=1)
        self.w=self.w-lr*dL_dw
        self.b=self.b-lr*dL_db
        dL_dA=np.matmul(grad,self.w.T)
        return dL_dA
    
    
class Reshape(Layer):
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape

    def forward(self, input):
        return np.reshape(input, self.output_shape)

    def backward(self, output_gradient, learning_rate):
        return np.reshape(output_gradient, self.input_shape)
    
    
class Sigmoid(Activation):
    def __init__(self):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        def sigmoid_prime(x):
            s = sigmoid(x)
            return s * (1 - s)

        super().__init__(sigmoid, sigmoid_prime)
        
class Softmax(Layer):
    def forward(self, input):
        tmp = np.exp(input)
        self.output = tmp / np.sum(tmp)
        return self.output
    
    def backward(self, output_gradient, learning_rate):
        # This version is faster than the one presented in the video
        n = np.size(self.output)
        print(n)
        return np.dot( (np.identity(n) - self.output.T) * self.output,output_gradient)
        # Original formula:
        # tmp = np.tile(self.output, n)
        # return np.dot(tmp * (np.identity(n) - np.transpose(tmp)), output_gradient)

class Tanh(Activation):
    def __init__(self):
        def tanh(x):
            return np.tanh(x)

        def tanh_prime(x):
            return 1 - np.tanh(x) ** 2

        super().__init__(tanh, tanh_prime)

class reLU(Layer):
    def forward(self, input):
        self.input = input
        return np.maximum(0, input)

    def backward(self, output_gradient, learning_rate):
        return np.multiply(output_gradient, self.input > 0)
    
    
def preprocess_data(x, y, limit, flag):
    x = x.reshape(len(x), 1, 28, 28)
    x = x.astype("float32") / 255
    if flag==True:
        zero_index = np.where(y == 0)[0][:limit]
        one_index = np.where(y == 1)[0][:limit]
        all_indices = np.hstack((zero_index, one_index))
        all_indices = np.random.permutation(all_indices)
        x, y = x[all_indices], y[all_indices]
    y = to_categorical(y)
    if flag==True:
        y = y.reshape(len(y), 2)
    else:
        y = y.reshape(len(y), 10)
    return x[:limit], y[:limit]


def load_data(limit, flag=False):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, y_train = preprocess_data(x_train, y_train, limit, flag)
    x_test, y_test = preprocess_data(x_test, y_test, limit, flag)
    
    return x_train, y_train, x_test, y_test

# load MNIST from server, limit to 100 images per class since we're not training on GPU

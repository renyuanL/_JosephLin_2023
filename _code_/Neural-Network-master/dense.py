import numpy as np
from   layer import Layer

class Dense000(Layer):
    def __init__(self, input_size, output_size):
        
        self.weights= np.random.randn(
            output_size, input_size  
            # TODO: 這裡不妙，
            # 希望能夠改成 input_size, output_size
            # 但是這樣會導致 backward() 的 input_gradient 計算錯誤
            # 故暫時保留原本的寫法 
            )

        self.bias = np.random.randn(output_size, 1)

    def forward(self, input):
        self.input=  input
        self.output= self.weights @ self.input + self.bias
        # TODO: 這裡不妙，
        # 希望能夠改成 self.output= self.input @ self.weights + self.bias

        return self.output

    def backward(self, output_gradient, learning_rate):

        # TODO: 這裡不妙，
        # 還需要再研究一下，才能夠改成正確的寫法。 

        weights_gradient= np.dot(output_gradient, self.input.T)
        input_gradient=   np.dot(self.weights.T, output_gradient)
        self.weights -= learning_rate * weights_gradient
        self.bias    -= learning_rate * output_gradient
        
        return input_gradient

# ryEdit: 2021-03-01 10:00:00
# Path: _code_\Neural-Network-master\dense.py
# 使用 x @ W -> y 的寫法
# $$ \sum_{i} x_i * w_{i,j} -> y_j $$


class Dense(Layer):
    def __init__(self, input_size, output_size):
        self.weights= np.random.randn(
            input_size, 
            output_size  
            )
        self.bias= np.random.randn(
            1, 
            output_size)

    def forward(self, input):
        self.input=  np.array(input).reshape(1, -1) # a row vector
        self.output= self.input @ self.weights + self.bias
        return self.output

    def backward(self, output_gradient, learning_rate):

        weights_gradient= self.input.T    @ output_gradient
        bias_gradient=    output_gradient
        input_gradient=   output_gradient @ self.weights.T 
        
        self.weights -= weights_gradient * learning_rate
        self.bias    -= bias_gradient    * learning_rate
        
        return input_gradient
    
class Softmax(Layer):
    def forward(self, input):
        tmp = np.exp(input)
        self.output = tmp / np.sum(tmp)
        return self.output
    
    def backward(self, output_gradient, learning_rate):
        # This version is faster than the one presented in the video
        n= np.size(self.output)
        
        #input_gradient=   output_gradient @ self.weights.T 
        input_gradient= output_gradient @ ((np.identity(n) - self.output.T) * self.output)

        return input_gradient
    
        #return np.dot(
        #    (np.identity(n) - self.output.T) * self.output, 
        #    output_gradient)
    
        # 這段要小心，可能有錯，待驗證。
    
        # Original formula:
        # tmp = np.tile(self.output, n)
        # return np.dot(tmp * (np.identity(n) - np.transpose(tmp)), output_gradient)

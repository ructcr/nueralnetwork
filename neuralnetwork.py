import numpy as np 
import random 
class Network(object):
    def __init__(self,sizes):
        self.sizes=sizes
        self.weights=[np.random.randn(x,y) for x,y in zip(self.sizes[:-1],self.sizes[1:])]
        self.biases=[np.random.randn(x) for x in self.sizes[1:]]        
    def SGD(self,training_data,epochs,mini_batch_size,eta,test_data=None):
        if test_data:n_test=len(test_data)
        n=len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches=[training_data[k:k+mini_batch_size] for k in range(0,n,mini_batch_size)]
            for mini_batch in mini_batches:
                print('minibatch:',mini_batch)
                self.update_mini_batch(mini_batch,eta)
            if test_data:
                print ("Epoch {0}:{1}/{2}".format(j,self.evaluate(test_data),n_test))
            else:
                print ("Epoch {0} complete".format(j))
    def update_mini_batch(self,mini_batch,eta):        
        for x,y in mini_batch:
            layers=self.feedfoward(x)
            self.backprop(layers,y)
          
    def feedfoward(self,layer):
        temp_layer=layer 
        layers=[temp_layer]
        for b,w in zip(self.biases,self.weights):
            temp_layer=self.sigmoid(np.dot(temp_layer,w)+b)
            layers.append(temp_layer)
        return layers
    def backprop(self,layers,y):
        for i in range(self.weights)
       
    def cost_derivative(self,output_activations,y):
        return(output_activations-y)
    def evaluate(self,test_data):
        test_results=[(np.argmax(self.feedfoward(x)),y) for (x,y) in test_data]
        return sum(int(x==y) for (x,y) in test_results)
    def sigmoid(self,z):
        return 1.0/(1.0+np.exp(-z))
    def sigmoid_prime(self,z):
        return sigmoid(z)*(1-sigmoid(z))

training_data=[([0,0,1],1),([0,1,0],0),([1,0,0],1)]

inodes=3
onodes=2
mnodes=5
net=Network([inodes,mnodes,onodes])#构建神经网络的随机初始值

epochs=1
mini_batch=3
eta=0.3
net.SGD(training_data,epochs,mini_batch,eta)#用梯度下降法训练神经网络

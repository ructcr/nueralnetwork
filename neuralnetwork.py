import numpy as np 
import random 
class Network(object):
    def __init__(self,sizes):
        self.sizes=sizes        
        self.weights=[[[0.15,0.25],[0.20,0.30]],[[0.40,0.50],[0.45,0.55]]]
        self.biases=[[0.35,0.35],[0.60,0.60]]
    def SGD(self,training_data,epochs,mini_batch_size,eta,test_data=None):
        if test_data:n_test=len(test_data)
        n=len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches=[training_data[k:k+mini_batch_size] for k in range(0,n,mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch,eta)
            if test_data:
                print ("Epoch {0}:{1}/{2}".format(j,self.evaluate(test_data),n_test))
            else:
                print ("Epoch {0} complete".format(j))
    def update_mini_batch(self,mini_batch,eta):        
        for x,y in mini_batch:
            layers=self.feedfoward(x)
            print('output:',layers[2][0],layers[2][1])
            print("E:",((layers[2][0]-y[0])*(layers[2][0]-y[0])+(layers[2][1]-y[1])*(layers[2][1]-y[1]))/2)
            self.backprop(layers,y,eta)          
    def feedfoward(self,layer):
        temp_layer=layer 
        layers=[temp_layer]
        for b,w in zip(self.biases,self.weights):
            net=np.dot(temp_layer,w)+b            
            temp_layer=self.sigmoid(net)           
            layers.append(temp_layer)
        return layers
    def backprop(self,layers,y,eta):
        tweights=self.weights
        for i in range(len(self.weights)):
            for j in range(len(self.weights[i])):
                for k in range(len(self.weights[i][j])):
                    if (i==1): self.weights[i][j][k]-=eta*(layers[i+1][k]-y[k])*layers[i+1][k]*(1-layers[i+1][k])*layers[i][j]
                    if (i==0): 
                        for p in range(len(layers[i+2])):                               
                            self.weights[i][j][k]-=eta*(layers[i+2][p]-y[p])*layers[i+2][p]*(1-layers[i+2][p])*self.weights[i+1][k][p]*layers[i+1][k]*(1-layers[i+1][k])*layers[i][j]      
        for i in range(len(self.biases)):        
            for j in range(len(self.biases[i])):
                if(i==0):
                    for k in range(len(layers[i+2])):
                        self.biases[i][j]-=eta*(layers[i+2][k]-y[k])*layers[i+2][k]*(1-layers[i+2][k])*tweights[i+1][j][k]*layers[i+1][j]*(1-layers[i+1][j])
                if(i==1):
                    self.biases[i][j]-=eta*(layers[i+1][j]-y[j])*layers[i+1][j]*(1-layers[i+1][j])                
    def evaluate(self,test_data):
        test_results=[(np.argmax(self.feedfoward(x)),y) for (x,y) in test_data]
        return sum(int(x==y) for (x,y) in test_results)
    def sigmoid(self,z):
        return 1.0/(1.0+np.exp(-z))
    def sigmoid_prime(self,z):
        return sigmoid(z)*(1-sigmoid(z))
training_data=[([0.05,0.10],[0.01,0.99])]
input_nodes=2
output_nodes=2
mid_nodes=2
net=Network([input_nodes,mid_nodes,output_nodes])#构建神经网络的随机初始值
epochs=10000
mini_batch=1
eta=0.5
net.SGD(training_data,epochs,mini_batch,eta)#用梯度下降法训练神经网络
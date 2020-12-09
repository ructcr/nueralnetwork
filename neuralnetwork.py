import numpy as np 
import random 
from trainingdata import training_data
from testdata import test_data
class Network(object):
    def __init__(self,sizes):
        self.sizes=sizes        
        #self.weights=[np.random.randn(x,y) for x,y in zip(sizes[:-1],sizes[1:])]                
        #self.biases=[np.random.randn(x) for x in sizes[1:]]
        self.weights=np.array([[[0.15,0.25],[0.20,0.30]],[[0.40,0.50],[0.45,0.55]]])
        self.biases=np.array([[0.35,0.35],[0.60,0.60]])
    def mini_batch_SGD(self,training_data,epochs,mini_batch_size,eta,test_data=None):
        if test_data:n_test=len(test_data)
        n=len(training_data)        
        for j in range(epochs):
            counter=0
            random.shuffle(training_data)
            for k in range(0,n,mini_batch_size):
                mini_batch=training_data[k:k+mini_batch_size]
                self.update_mini_batch(mini_batch,eta)                
                #print('minibatch {0} complete'.format(counter))
                counter+=1
            if test_data:
                print ("Epoch {0}:{1}/{2}".format(j,self.evaluate(test_data),n_test))
            else:
                print ("Epoch {0} complete".format(j))
    def update_mini_batch(self,mini_batch,eta):        
        s_weights=[np.zeros(x.shape) for x in self.weights]
        s_biases=[np.zeros(x.shape) for x in self.biases]
        for x,y in mini_batch:                                
            layers=self.feedfoward(x)                                           
            t_weights,t_biases=self.backprop(layers,y)        
            s_weights=[sw+tw for sw,tw in zip(s_weights,t_weights)]            
            s_biases=[sb+tb for sb,tb in zip(s_biases,t_biases)]        
        self.weights=[w-sw*eta/len(mini_batch) for w,sw in zip(self.weights,s_weights)]        
        #self.biases=[b-sb*eta/len(mini_batch) for b,sb in zip(self.biases,s_biases)]      
        print(self.weights)          
    def feedfoward(self,layer):
        temp_layer=layer 
        layers=[temp_layer]
        for b,w in zip(self.biases,self.weights):
            net=np.dot(temp_layer,w)+b                                 
            temp_layer=self.sigmoid(net)             
            layers.append(temp_layer)
        return layers
    def backprop(self,layers,y):
        t_weights=[np.zeros(x.shape) for x in self.weights]
        t_biases=[np.zeros(x.shape) for x in self.biases]        
        dnetO=(layers[-1]-y)*(layers[-1]*(1-layers[-1]))        
        t_weights[-1]=np.dot(layers[-2].reshape(len(layers[-2]),1),dnetO.reshape(1,len(layers[-1])))
        t_biases[-1]=dnetO
        for i in range(2,len(self.sizes)):
            dnetO=(dnetO.dot(self.weights[-i+1]).T)*(layers[-i]*(1-layers[-i]))            
            t_weights[-i]=(layers[-i-1].reshape(len(layers[-i-1]),1)).dot(dnetO.reshape(1,len(layers[-i])))            
            t_biases[-i]=dnetO
        return t_weights,t_biases        
    def evaluate(self,test_data):
        test_results=[(np.argmax(self.feedfoward(x)[-1]),y) for (x,y) in test_data]
        return sum(int(x==y) for (x,y) in test_results)
    def sigmoid(self,z):
        return 1.0/(1.0+np.exp(-z))
    
#training_data=training_data().get_data()
#test_data=test_data().get_data()
training_data=np.array([[[0.05,0.1],[0.01,0.99]]])
input_nodes=2
mid_nodes=2
output_nodes=2
net=Network([input_nodes,mid_nodes,output_nodes])#构建神经网络的随机初始值

epochs=2
mini_batch_size=1
eta=0.5
net.mini_batch_SGD(training_data,epochs,mini_batch_size,eta)#用梯度下降法训练神经网络
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
        nabla_b=[np.zeros(b.shape) for b in self.biases]
        nabla_w=[np.zeros(w.shape) for w in self.weights]
        #print(nabla_b,'\n',nabla_w)
        for x,y in mini_batch:
            #print('x:',x,' y:',y)
            delta_nabla_b,delta_nabla_w=self.backprop(x,y)
            #print('deltas:',delta_nabla_b,delta_nabla_w)
            """ nabla_b=[nb+dnb for nb,dnb in zip(nabla_b,delta_nabla_b)]
            nabla_w=[nw+dnw for nw,dnw in zip(nabla_w,delta_nabla_w)]
            self.weights=[w-(eta/len(mini_batch))*nw for w,nw in zip(self.weights,nabla_w)]
            self.biases=[b-(eta/len(mini_batch))*nb for b,nb in zip(self.biases,nabla_b)] """
    def feedfoward(self,a):
        for b,w in zip(self.biases,self.weights):
            a=self.sigmoid(np.dot(a,w)+b)
        return a
    def backprop(self,x,y):
        nabla_b=[np.zeros(b.shape) for b in self.biases]
        nabla_w=[np.zeros(w.shape) for w in self.weights]
        activation=x
        activations=[x]
        zs=[]
        for b,w in zip(self.biases,self.weights):
            #print(b,'\n\n',w,'\n\n\n')
            z=np.dot(activation,w)+b
            zs.append[z]
            activation=self.sigmoid(z)
            activations.append(activation)             
            #print(activation)
        delta=self.cost_derivative(activations[-1],y)*self.sigmoid_prime(zs[-1])
        nabla_b[-1]=delta
        nabla_w[-1]=np.dot(delta,activations[-2].transpose())
        return(nabla_b,nabla_w)
    def cost_derivative(self,output_activations,y):
        return(output_activations-y)
    def evaluate(self,test_data):
        test_results=[(np.argmax(self.feedfoward(x)),y) for (x,y) in test_data]
        return sum(int(x==y) for (x,y) in test_results)
    def sigmoid(self,z):
        return 1.0/(1.0+np.exp(-z))
    def sigmoid_prime(self,z):
        return sigmoid(z)*(1-sigmoid(z))

net=Network([3,5,2])
net.SGD([([0,0,1],1),([0,1,0],0),([1,0,0],1)],1,3,0.3)

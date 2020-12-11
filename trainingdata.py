import numpy as np

class training_data():
    def __init__(self):        
        with open('train-labels.idx1-ubyte','rb') as f:
            magic_number=int(f.read(4).hex(),16)
            items_number=int(f.read(4).hex(),16)
            self.labels=np.zeros((60000,10),dtype=np.int)            
            tlabel=np.frombuffer(f.read(),dtype=np.uint8).copy()
            for i in range(60000):
                self.labels[i][tlabel[i]]=1
        with open('train-images.idx3-ubyte','rb') as f:      
            magic_number=int(f.read(4).hex(),16)
            items_number=int(f.read(4).hex(),16)
            rows=int(f.read(4).hex(),16)
            coloumns=int(f.read(4).hex(),16)            
            self.datas=np.frombuffer(f.read(),dtype=np.uint8).reshape(60000,784).copy()
            self.datas[self.datas<30]=0
            self.datas[self.datas>=30]=255
    def get_data(self):                                
        data=[]        
        for i in range(60000):
            tdata=[]
            tdata.append(self.datas[i])
            tdata.append(self.labels[i])           
            data.append(tdata)
        return data    
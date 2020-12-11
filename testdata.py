import numpy as np

class test_data():
    def __init__(self):        
        with open('t10k-labels.idx1-ubyte','rb') as f:
            magic_number=int(f.read(4).hex(),16)
            items_number=int(f.read(4).hex(),16)
            self.labels=np.frombuffer(f.read(),dtype=np.uint8).copy()            
        with open('t10k-images.idx3-ubyte','rb') as f:      
            magic_number=int(f.read(4).hex(),16)
            items_number=int(f.read(4).hex(),16)
            rows=int(f.read(4).hex(),16)
            coloumns=int(f.read(4).hex(),16)            
            self.datas=np.frombuffer(f.read(),dtype=np.uint8).reshape(10000,784).copy()
            self.datas[self.datas<30]=0
            self.datas[self.datas>=30]=255
    def get_data(self):                                
        data=[]        
        for i in range(10000):
            tdata=[]
            tdata.append(self.datas[i])
            tdata.append(self.labels[i])           
            data.append(tdata)
        return data    
labels=[]
with open('train-labels.idx1-ubyte','rb') as f:
    magic_number=int(f.read(4).hex(),16)
    items_number=int(f.read(4).hex(),16)
    print(magic_number,items_number)
    for i in range(60000):
        labels.append(int(f.read(1).hex(),16))
print(len(labels))
datas=[]
with open('train-images.idx3-ubyte','rb') as f:
    magic_number=int(f.read(4).hex(),16)
    items_number=int(f.read(4).hex(),16)
    rows=int(f.read(4).hex(),16)
    coloumns=int(f.read(4).hex(),16)
    print(magic_number,items_number,rows,coloumns)
    for i in range(60000):
        datas.append([])
        for j in range(28):
            datas[i].append([])
            for k in range(28):
                datas[i][j].append(int(f.read(1).hex(),16))
print(len(datas))
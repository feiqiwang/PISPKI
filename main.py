from bindingPlaceDetector_NN import bindingPlaceDetector
from data_iter_molcl import ligandDataset
import numpy as np
import torch
from pytorchtools import EarlyStopping
import random
import time
import torch.utils.data as Data

device = torch.device("cuda")
use_GPU = torch.cuda.is_available()
use_GPU = False


epoch_num = 50

#Early stopping
patience = 5
early_stopping = EarlyStopping(patience, verbose = True)

#batch size is for both of positive samples and negative samples.
batch_size = 16
#same with batch size.
turn_size = 2048

batch_size_val = 16
turn_size_val = 256

net = bindingPlaceDetector()
loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

ProteinAC = "TPK"
path = "./dataset/" + ProteinAC
dataset = ligandDataset(path,mode="bpd_set")

acc_result_filename = "acc_"+ProteinAC+"_"+"".join(map(str,list(time.localtime()[0:5])))+".csv"
acc_file = open(acc_result_filename,'w')
acc_file.write(ProteinAC+",loss,train_acc,valid_acc\n")


def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    for X,s, y in data_iter:
        if (use_GPU):
            y = y.to(device)

        acc_sum += (net(X,s).argmax(dim=1) == y).float().sum().item()
        n += y.shape[0]

    return acc_sum / n

def train_softmax(net, dataset, loss_function, epochs_num, turn_size, batch_size, optimizer):
    if(use_GPU):
        print("This softmax training function is on",device)

    start_time = time.time()

    
    
    for epoch in range(epochs_num):
        train_l_sum, train_acc_sum, n =0.0, 0.0, 0
        
        train_iter = dataset.DataLoader(batch_size,turn_size)
        val_iter = dataset.DataLoader_val(batch_size_val,turn_size_val)

        net.train()
        for X, s, y in train_iter:
            print(".",end=''),
            if (use_GPU):
                y = y.to(device)
            optimizer.zero_grad()
            y_hat = net(X,s)
            l = loss(y_hat, y).sum()

            l.backward()

            optimizer.step()

            train_l_sum += l.item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
            n += y.shape[0]

        net.eval()
        val_l,val_acc,val_n = 0.0,0.0,0

        for X, s, y in val_iter:
            if (use_GPU):
                y = y.to(device)
            y_hat = net(X,s)
            val_l += loss(y_hat, y).sum().item()
            val_acc += (y_hat.argmax(dim=1) == y).sum().item()
            val_n += y.shape[0]


        print("Total time cost",int((time.time()-start_time)/60),"minutes")
        print('epoch %d, loss %.4f, train acc %.3f, val_loss %.4f, val acc %.3f' % (epoch + 1, train_l_sum / n, train_acc_sum / n, val_l/val_n, val_acc/val_n))
        acc_file.write("epoch"+str(epoch+1)+","+str(train_l_sum/n)+","+str(100*train_acc_sum/n)+","+str(100*val_acc/val_n)+"\n")
        early_stopping(val_l,net)
        if early_stopping.early_stop:
            print("Early stopping!")
            break
    net.load_state_dict(torch.load('checkpoint.pt'))

    
    
    print()
    print("Start to evaluate model performance by using test dataset.")

    eval_iter = dataset.DataLoader_test(batch_size)
    conf_iter = dataset.DataLoader_confuse(batch_size,0.5)
    
    acc_sum,n=0.0,0
    
    net.eval()
    for X, s, y in eval_iter:
        print(".",end='')
        if (use_GPU):
            y = y.to(device)
        
        acc_sum += (net(X,s).argmax(dim=1) == y).float().sum().item()
        n += y.shape[0]
        

    print("Model performance(test): %.4f",acc_sum/n)
    acc_file.write("Model performance, "+str(acc_sum/n))
    acc_file.write("Total time cost(min),"+str(int((time.time()-start_time)/60)))

    print()
    print("Start to evaluate model performance by using confused dataset.")
    acc_sum,n=0.0,0
    for X, s, y in conf_iter:
        print(".",end='')
        if (use_GPU):
            y = y.to(device)
        acc_sum += (net(X,s).argmax(dim=1) == y).float().sum().item()
        n += y.shape[0]

    print("Model performance(confused): %.4f",acc_sum/n)
    acc_file.write("Model performance(confused), "+str(acc_sum/n))

train_softmax(net,dataset,loss,epoch_num,turn_size, batch_size,optimizer)

acc_file.close()

print("Start to save the network...")
torch.save(net,"net_"+ProteinAC+"_"+"".join(map(str,list(time.localtime()[0:5]))))
print("End")








    

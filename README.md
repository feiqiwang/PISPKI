# PISPKI
[Correspondence: wangfeiqi@kuicr.kyoto-u.ac.jp]\
The PISPKI model is a PyTorch-based program developed for the prediction of interaction sites of protein kinase inhibitors, which is a Weisfiler-Lehman algorithm-based graph neural network attached with a novel module (i.e., WL Box). 

## Requirement

The program is supported by Python v.3.8.\
Please install latest [PyTorch](https://pytorch.org/).\
[Optional]: If you want to apply early-stopping (We applied it in our research), please visit this [page](https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py) by Bjarten and install it.

## Usage
The `PISPKI` model is compatible with the input of *mol2* file format. Collect *mol2* files of inhibitors of a kinase and put them into a folder. 

Write your folder name and its path in `main.py` as follow:
```python
ProteinAC = "TPK"
path = "./dataset/" + ProteinAC
```
Regardless of how much inhibitor data you give, the program can expand it to 2,560 positive and 2,560 negative datasets.

Due to the copyright problem, we do not offer kinase inhibitor data as an example.

Then, you can run `main.py` directly if you prefer our setting of the model.

Or you can custom training parameters in the `main.py` file by following codes:
```python
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
```

You can build a special PISPKI model, but we do not recommend you do it if you are not familiar with *PyTorch*.\
Find bindingPlaceDetector class from the `bindingPlaceDetector_NN.py` file and custom parameters of the PISPKI model by the following code:
```python
#The layer number of the first WL Box.
conv_num_1 = 2
#The layer number of the second WL Box.
conv_num_2 = 2
#The time step number of the first WL Box.
conv_times_1 = 2
#The time step number of the second WL Box.
conv_times_2 = 2

#The out channel number of the first convolutional layer.
cnn_out_channel = 2
#The out channel number of the second convolutional layer.
cnn_out_channel_2 = 5
#Kernel size of both first and second convolutional layers
cnn_kernel = 3

#SPP stage number of the output from WL Boxes.
spp_len = 10
#SPP stage number of the output from conovlutional layers
spp_len_stru = 3

#Dropout rate of the dense layer.
drop_p= 0.05
#The neuron number of each layer in dense network (except input and output layer).
hidden_neuron_num = 2000
```
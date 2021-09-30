import torch
import math
import numpy as np
from torch import nn

device = torch.device("cuda")
use_GPU = torch.cuda.is_available()
use_GPU = False

one_hot_encoding_size = 35

class weisfeiler_lehman_conv(nn.Module):

    def __init__(self, out_channels, convoTimes, kernel_size=16):
        super(weisfeiler_lehman_conv,self).__init__()
        self.out_channels =  out_channels
        self.convoTimes = convoTimes
        self.kernels = nn.ParameterDict({'kernel '+str(i):nn.Parameter(torch.rand(convoTimes,kernel_size))for i in range(out_channels)})

    def WLC(self,labelsList,ligand_structure):
        newLabelsList = labelsList.repeat(self.out_channels,1,1)
        for neurals in range(self.out_channels):
            for convo_time in range(self.convoTimes):
                cal_list = newLabelsList[neurals].clone()
                for index in range(newLabelsList.shape[1]):
                    for struc_index in range(ligand_structure.shape[1]):
                        if ligand_structure[index][struc_index]>= 1:
                            newLabelsList[neurals][index] += cal_list[struc_index]*self.kernels['kernel ' + str(neurals)][convo_time]

        return newLabelsList


    def forward(self,labelsList,ligand_structure):
        if (str(type(labelsList)) == "<class 'list'>"):
            output = []
            for i in range(len(labelsList)):
                result = self.WLC(labelsList[i],ligand_structure[i])
                output.append(result.clone())
            return output
        elif (len(labelsList.shape) == 2):
            return self.WLC(labelsList,ligand_structure)
        elif (len(labelsList.shape) == 3 and len(ligand_structure.shape) == 2):#second times apply
            output = []
            for i in range(labelsList.shape[0]):
                result = self.WLC(labelsList[i],ligand_structure)
                output.append(result.clone())
            return torch.cat(output,dim=0)
        elif (len(labelsList.shape) == 3 and len(ligand_structure.shape) == 3):#batch processing
            output = []
            for i in range(labelsList.shape[0]):
                result = self.WLC(labelsList[i],ligand_structure[i])
                output.append(torch.unsqueeze(result.clone(),0))
            return torch.cat(output,dim=0)
                
        else:
            print("Input size error.")
            return None


class weisfeiler_lehman_conv_switch(nn.Module):

    def __init__(self, out_channels, convoTimes, kernel_size=(4,one_hot_encoding_size)):
        super(weisfeiler_lehman_conv_switch,self).__init__()
        self.out_channels =  out_channels
        self.convoTimes = convoTimes
        self.kernels = nn.ParameterDict({'kernel '+str(i):nn.Parameter(torch.rand(convoTimes,kernel_size[0],kernel_size[1]))for i in range(out_channels)})

    def WLC(self,labelsList,ligand_structure):
        newLabelsList = labelsList.repeat(self.out_channels,1,1)
        for neurals in range(self.out_channels):
            for convo_time in range(self.convoTimes):
                cal_list = newLabelsList[neurals].clone()
                for index in range(newLabelsList.shape[1]):
                    for struc_index in range(ligand_structure.shape[1]):
                        if ligand_structure[index][struc_index]>= 1:
                            newLabelsList[neurals][index] += cal_list[struc_index]*self.kernels['kernel ' + str(neurals)][convo_time][int(ligand_structure[index][struc_index])-1]

        return newLabelsList


    def forward(self,labelsList,ligand_structure):
        if (str(type(labelsList)) == "<class 'list'>"):
            output = []
            for i in range(len(labelsList)):
                result = self.WLC(labelsList[i],ligand_structure[i])
                output.append(result.clone())
            return output
        elif (len(labelsList.shape) == 2):
            return self.WLC(labelsList,ligand_structure)
        elif (len(labelsList.shape) == 3 and len(ligand_structure.shape) == 2):#second times apply
            output = []
            for i in range(labelsList.shape[0]):
                result = self.WLC(labelsList[i],ligand_structure)
                output.append(result.clone())
            return torch.cat(output,dim=0)
        elif (len(labelsList.shape) == 3 and len(ligand_structure.shape) == 3):#batch processing
            output = []
            for i in range(labelsList.shape[0]):
                result = self.WLC(labelsList[i],ligand_structure[i])
                output.append(torch.unsqueeze(result.clone(),0))
            return torch.cat(output,dim=0)
                
        else:
            print("Input size error.")
            return None
                    

class spatial_pyramid_pool(nn.Module):
    def __init__(self, levels, poolType):
        # poolType could be max_pool, avg_pool, lp_pool
        super(spatial_pyramid_pool,self).__init__()
        self.levels = levels
        self.poolType = poolType

    def spp(self,X):
        spp_results=[]
        for data in X:
            spp_res=[]
            for pool_prop in range(1,self.levels+1):
                pool_size = [int(math.ceil(data.shape[0]/pool_prop)),int(math.ceil(data.shape[1]/pool_prop))]
                pool_pad = [pool_size[0]*pool_prop - data.shape[0], pool_size[1]*pool_prop - data.shape[1]]

                if (pool_pad[0]*2 > pool_size[0]):
                    pool_pad[0] = int(math.floor(pool_size[0]/2))

                if (pool_pad[1]*2 > pool_size[1]):
                    pool_pad[1] = int(math.floor(pool_size[1]/2))


  
                if (self.poolType == "max_pool"):
                    pool = nn.MaxPool2d(pool_size,pool_size,pool_pad)
                elif (self.poolType == "avg_pool"):
                    pool = nn.AvgPool2d(pool_size,pool_size,pool_pad)
                elif (self.poolType == "lp_pool"):
                    pool = nn.LPPool2d(pool_size,pool_size,pool_pad)
                else:
                    print("Wrong pool name.")
                pool_result = pool(data.view(1,1,data.shape[0],data.shape[1]))
                #spp_res.append(pool_result.clone().view(pool_prop*pool_prop))            
                spp_res.append(pool_result.clone().view(pool_result.shape[2]*pool_result.shape[3]))
                spp_res.append(torch.zeros(pool_prop*pool_prop-pool_result.shape[2]*pool_result.shape[3]))
            spp_res_tensor = torch.cat(spp_res, dim=0)
            spp_results.append(spp_res_tensor.clone().view(1,spp_res_tensor.nelement()))
        spp_results_tensor = torch.cat(spp_results, dim=0)

        return spp_results_tensor

    def forward(self,X):
        if (str(type(X)) == "<class 'list'>"):
            output = []
            for i in range(len(X)):
                result = self.spp(X[i])
                output.append(torch.unsqueeze(result.clone(),0))
            return torch.cat(output,dim=0)
        elif (len(X.shape) == 3):
            return self.spp(X)
        elif (len(X.shape) == 4):#batch processing
            output = []
            for i in range(X.shape[0]):
                result = self.spp(X[i])
                output.append(torch.unsqueeze(result.clone(),0))
            return torch.cat(output,dim=0)
        else:
            print("Input size error.")
            return None


class primary_capsule(nn.Module):

    def __init__(self, in_channels, out_channels, capsNum):
        super(primary_capsule,self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.capsNum = capsNum

        self.weight = nn.Parameter(0.001*torch.rand(out_channels, in_channels[0], capsNum, in_channels[1]))

    def forward(self,X):
        output = []
        for index in range(X.shape[0]):
            labels = []
            for i in range(self.out_channels):
                features = []
                for j in range(self.in_channels[0]):
                    for k in range(self.capsNum):
                        features.append(torch.unsqueeze(X[index][j]*self.weight[i][j][k],0))
                labels.append(torch.unsqueeze(squash(torch.sum(torch.cat(features,dim=0),dim=0)),0))
            output.append(torch.unsqueeze(torch.cat(labels,dim=0),0))
        

        return torch.cat(output,dim=0)

        



class activation_function(nn.Module):
    def __init__(self,af_type):
        super(activation_function,self).__init__()
        self.af_type = af_type

    def forward(self,X):
        output = []
        if (self.af_type == "LeakyReLU"):
            layer = nn.LeakyReLU()
        elif (self.af_type == "Tanh"):
            layer = nn.Tanh()
        elif (self.af_type == "Sigmoid"):
            layer = nn.Sigmoid()
        for i in range(len(X)):
            result = layer(X[i])
            output.append(result.clone())
        return output


class bindingPlaceDetector(nn.Module):
    def __init__(self):
        super(bindingPlaceDetector,self).__init__()

        conv_num_1 = 2
        conv_num_2 = 2
        conv_times_1 = 2
        conv_times_2 = 2

        cnn_rate = 3

        cnn_out_channel = 2
        cnn_out_channel_2 = 5
        cnn_kernel = 3

        spp_len = 10
        spp_len_stru = 3
        
        drop_p= 0.05

        hidden_neuron_num = 2000
 

        #################
        linear_input = 0 #decided on spp_len
        for i in range(spp_len):
            linear_input+=(i+1)*(i+1)
        linear_input = linear_input*conv_num_1*conv_num_2#*cnn_rate
        for i in range(spp_len_stru):
            linear_input+=(i+1)*(i+1)*cnn_out_channel_2
        #################
        
        self.conv_model = weisfeiler_lehman_conv_switch(conv_num_1, conv_times_1)
        self.af = nn.LeakyReLU()
        self.conv_model_2 = weisfeiler_lehman_conv_switch(conv_num_2, conv_times_2)
        self.pool_model = nn.Sequential(
            #nn.LeakyReLU(),

            #nn.Conv2d(conv_num_1*conv_num_2, conv_num_1*conv_num_2*cnn_rate, cnn_kernel),

            nn.LeakyReLU(),
         
            spatial_pyramid_pool(spp_len, "max_pool")       
            )

        self.cnn_model = nn.Sequential(
            nn.Conv2d(1,cnn_out_channel,cnn_kernel),
            nn.LeakyReLU(),
            nn.Conv2d(cnn_out_channel,cnn_out_channel_2,cnn_kernel),
            nn.LeakyReLU(),
            spatial_pyramid_pool(spp_len_stru, "avg_pool"))


        self.fc_model = nn.Sequential(
            nn.Linear(linear_input,hidden_neuron_num),
            nn.LeakyReLU(),
            nn.Dropout(drop_p),
            nn.Linear(hidden_neuron_num,hidden_neuron_num),
            nn.LeakyReLU(),
            nn.Dropout(drop_p),
            nn.Linear(hidden_neuron_num,hidden_neuron_num),
            nn.LeakyReLU(),
            nn.Dropout(drop_p),
            nn.Linear(hidden_neuron_num,hidden_neuron_num),
            nn.LeakyReLU(),
            nn.Dropout(drop_p),
            nn.Linear(hidden_neuron_num,2),
            )

        #####Initial network weights#####
        print("------------------Initial network parameter by Xavier--------------------")
        print(self)
        print("-------------------------------------------------------------------------")
        
        for m in self.modules():
            if isinstance(m, weisfeiler_lehman_conv_switch):           
                for k in m.kernels:
                    nn.init.xavier_uniform_(m.kernels[k],gain=1)
            elif isinstance(m, nn.Conv2d):              
                nn.init.xavier_normal_(m.weight,gain=1)
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
                nn.init.xavier_normal_(m.weight,gain=1)
        #################################
                

        if(use_GPU):
            self.fc_model = self.fc_model.to(device)
            
    def forward(self,labelsList,ligand_structure):
        feature = []
        structure = []
        for i in range(len(labelsList)):
            conv_result_1 = self.af(self.conv_model(labelsList[i],ligand_structure[i]))
            conv_result_2 = self.conv_model_2(conv_result_1,ligand_structure[i])
            pool_result = self.pool_model(conv_result_2)

            structure_result = self.cnn_model(ligand_structure[i].view(1,1,ligand_structure[i].shape[0],ligand_structure[i].shape[1]))  
            feature.append(torch.unsqueeze(pool_result.clone(),0))
            #feature.append(pool_result)
            structure.append(torch.unsqueeze(structure_result.clone(),0))

        feature = torch.cat(feature,dim=0)
        structure = torch.cat(structure,dim=0)
        if (use_GPU):
            feature = feature.to(device)
            structure = structure.to(device)
        #return self.caps_model(feature).norm(dim=-1)
        #feature = self.pool_model(self.conv_model(labelsList,ligand_structure))
        if len(feature.shape) == 2:
            #return self.fc_model(feature.view(-1))
            return self.fc_model(torch.cat((feature.view(-1),structure.view(-1)),1))
        elif len(feature.shape) == 3:
            #return self.fc_model(feature.view(feature.shape[0],-1))
            return self.fc_model(torch.cat((feature.view(feature.shape[0],-1),structure.view(structure.shape[0],-1)),1))


        
def squash(X,axis=-1):
    norm = torch.norm(X,p=2,dim=axis,keepdim=True)
    scale = norm**2/(1 + norm**2)/(norm + 1e-8)
    return scale * X

#X=torch.rand(4,15)
#net=capsule_network([4,15],2,8)
#net(X)

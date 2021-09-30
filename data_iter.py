from torch.utils.data import Dataset
from torch import nn
import torch
import random
import os
import math

class ligandDataset(Dataset):
    def __init__(self, path, mode="bpd"):
        super(ligandDataset,self).__init__

###################################################################
        print("Start to generate original dataset for "+path.split('/')[-1]+"...")
        #List saving label feature
        self.ligand_feature_list_positive = []
        self.ligand_feature_list_negative = []
        #List saving structure
        self.ligand_structure_list_positive = []
        self.ligand_structure_list_negative = []
###################################################################


        if (os.path.isdir(path + "/" + os.listdir(path)[0] + "/" + os.listdir(path+"/"+os.listdir(path)[0])[0])):
            for pAC in os.listdir(path):
                self.ProteinAc = pAC
                self.creatDataset(path+"/"+pAC)
        else:         
            self.ProteinAc = path.split('/')[-1]
            self.creatDataset(path)
        
        if(mode == "cnn" or mode == "fcnn" or mode == "svm"):
            self.fiter()
        elif(mode[0:3] == "bpd"):           
            if (len(self.ligand_feature_list_positive) < 2560 or len(self.ligand_feature_list_negative) < 2560): #2048+256+256
                print("Due to the limited dataset, start the expander program......")
                self.expandDataset()
            self.splitDataset()
        else:
            print("Mode name error.")

        self.prop = 1
        print("The proportion of positive and negative data is "+str(self.prop)+".")



    def creatDataset(self,path):
        one_hot_size = 35

        for ligand_index in os.listdir(path):
            mol_file = open(path + "/" + ligand_index + "/ligand.mol2")
            bind_file = open(path + "/" + ligand_index + "/EXPORTINTS=" + ligand_index.replace("\n",""))
            mol_line = mol_file.readline()
            bind_line = bind_file.readline()

            #Get the atom numer
            mol_line = mol_file.readline()
            mol_line = mol_file.readline()
            mol_line = self.strSplit(mol_line)
            label_tensor = torch.zeros(int(mol_line[0]),one_hot_size)
            structure_tensor = torch.zeros(int(mol_line[0]),int(mol_line[0]))
            mol_line = mol_file.readline()
            #### #### #### ####
            
            while mol_line:
                if mol_line == "@<TRIPOS>ATOM\n":
                    mol_line = mol_file.readline()
                    while mol_line:
                        if (mol_line[0]=="@"):
                            break
                        mol_line = mol_line = self.strSplit(mol_line)

                        #write one-hot encoding
                        label_tensor[int(mol_line[0])-1][int(self.labelIden(mol_line[5]))] = 1

                        #Also label adjancy matrix but in minus
                        structure_tensor[int(mol_line[0])-1][int(mol_line[0])-1] = 0 - int(self.labelIden(mol_line[5])) - 1

                        
                        mol_line = mol_file.readline()
                        
                    
                if mol_line == "@<TRIPOS>BOND\n":
                     mol_line = mol_file.readline()
                     while mol_line:
                        if (mol_line[0]=="@"):
                            break
                        mol_line = mol_line = self.strSplit(mol_line)

                        #write structure
                        if (mol_line[3]=="2"):
                            structure_tensor[int(mol_line[1])-1][int(mol_line[2])-1] = 2
                            structure_tensor[int(mol_line[2])-1][int(mol_line[1])-1] = 2
                        elif (mol_line[3]=="ar"):
                            structure_tensor[int(mol_line[1])-1][int(mol_line[2])-1] = 3
                            structure_tensor[int(mol_line[2])-1][int(mol_line[1])-1] = 3
                        elif (mol_line[3]=="am"):
                            structure_tensor[int(mol_line[1])-1][int(mol_line[2])-1] = 4
                            structure_tensor[int(mol_line[2])-1][int(mol_line[1])-1] = 4
                        else:
                            structure_tensor[int(mol_line[1])-1][int(mol_line[2])-1] = 1
                            structure_tensor[int(mol_line[2])-1][int(mol_line[1])-1] = 1
                        
                        mol_line = mol_file.readline()


                mol_line = mol_file.readline()
            mol_file.close()

            bind_index=[]
            while bind_line:
                if (bind_line[0] == '#'):
                    bind_line=bind_file.readline()
                    while bind_line:
                        bind_line = self.strSplit(bind_line)
                        if bind_line[1] != "NULL" and int(bind_line[1]) not in bind_index:
                            bind_index.append(int(bind_line[1]))

                        bind_line = bind_file.readline()

                bind_line = bind_file.readline()

            ########Create orignal data iter#########
                ####Highlight method = (+1000)
            for i in range(label_tensor.shape[0]):
                if i+1 in bind_index:
                    self.ligand_feature_list_positive.append(label_tensor.clone())
                    self.ligand_structure_list_positive.append(structure_tensor.clone())
                    for j in range(self.ligand_feature_list_positive[-1].shape[1]):
                        if self.ligand_feature_list_positive[-1][i][j]!= 0:
                            self.ligand_feature_list_positive[-1][i][j] += 1000
                            break
                else:
                    self.ligand_feature_list_negative.append(label_tensor.clone())
                    self.ligand_structure_list_negative.append(structure_tensor.clone())
                    for j in range(self.ligand_feature_list_negative[-1].shape[1]):
                        if self.ligand_feature_list_negative[-1][i][j]!= 0:
                            self.ligand_feature_list_negative[-1][i][j] += 1000
                            break                    
            

        if (len(self.ligand_feature_list_positive) == len(self.ligand_structure_list_positive)):
            print("Finish generating the dataset from "+self.ProteinAc+".")
            print("The positive original dataset quality is " + str(len(self.ligand_feature_list_positive)) + ".")
            print("The negative original dataset quality is " + str(len(self.ligand_feature_list_negative)) + ".")
        else:
            print("Size error.")
        
        return 0



    def fiter(self):
        #dataset for svm
        self.ligand_dataset_svm_positive = []

        self.ligand_dataset_svm_negative = []

        #dataset for fcnn
        self.ligand_dataset_fcnn_positive = []

        self.ligand_dataset_fcnn_negative = []

        max_atom_num = 0

        
        for structure in self.ligand_structure_list_positive:
            if structure.shape[0] > max_atom_num:
                max_atom_num = structure.shape[0]

        for structure in self.ligand_structure_list_negative:
            if structure.shape[0] > max_atom_num:
                max_atom_num = structure.shape[0]
        print("The maximum atom number size of this database is "+str(max_atom_num)+".")
        print("Start to create dataset for Fully Connected Neural Network & Convolutional Nerual Network & Support Vector Machine...")
        for feature,structure in zip(self.ligand_feature_list_positive, self.ligand_structure_list_positive):
            pad_zero = nn.ZeroPad2d(padding=(0,max_atom_num-structure.shape[0],0,max_atom_num-structure.shape[1]))
            self.ligand_dataset_fcnn_positive.append(torch.unsqueeze(pad_zero(structure.clone()),0))
            for i in range(feature.shape[0]):
                for j in range(feature.shape[1]):
                    if feature[i][j] == 1:
                        self.ligand_dataset_fcnn_positive[-1][0][i][i] = j+1
                        break
                    elif feature[i][j] == 101:
                        self.ligand_dataset_fcnn_positive[-1][0][i][i] = j+101
                        break
            self.ligand_dataset_svm_positive.append((self.ligand_dataset_fcnn_positive[-1][0].view(max_atom_num*max_atom_num)).tolist())
            

        for feature,structure in zip(self.ligand_feature_list_negative, self.ligand_structure_list_negative):
            pad_zero = nn.ZeroPad2d(padding=(0,max_atom_num-structure.shape[0],0,max_atom_num-structure.shape[1]))
            self.ligand_dataset_fcnn_negative.append(torch.unsqueeze(pad_zero(structure.clone()),0))
            for i in range(feature.shape[0]):
                for j in range(feature.shape[1]):
                    if feature[i][j] == 1:
                        self.ligand_dataset_fcnn_negative[-1][0][i][i] = j+1
                        break
                    elif feature[i][j] == 101:
                        self.ligand_dataset_fcnn_negative[-1][0][i][i] = j+101
                        break
            self.ligand_dataset_svm_negative.append((self.ligand_dataset_fcnn_negative[-1][0].view(max_atom_num*max_atom_num)).tolist())
    
       
        

        
        

        
            




    
    def expandDataset(self):
        if (len(self.ligand_feature_list_positive) < 2560): #positive
            expand_rate = int(math.ceil(2560 / len(self.ligand_feature_list_positive)))

            self.ligand_feature_list_positive = self.ligand_feature_list_positive * expand_rate
            self.ligand_structure_list_positive = self.ligand_structure_list_positive * expand_rate
            
            seed = random.randint(1,1000)
            for i in range(int(len(self.ligand_structure_list_positive)/expand_rate),len(self.ligand_structure_list_positive)):              
                torch.manual_seed(seed)
                self.ligand_feature_list_positive[i] = self.ligand_feature_list_positive[i][torch.randperm(self.ligand_feature_list_positive[i].size(0))]
                torch.manual_seed(seed)
                self.ligand_structure_list_positive[i] = self.ligand_structure_list_positive[i][torch.randperm(self.ligand_structure_list_positive[i].size(0))]
                torch.manual_seed(seed)
                self.ligand_structure_list_positive[i] = self.ligand_structure_list_positive[i][:,torch.randperm(self.ligand_structure_list_positive[i].size(1))]
                if (i%100 == 0):##Change the seed
                    seed = random.randint(1,1000)

        
        if (len(self.ligand_feature_list_negative) < 2560):#negative
            expand_rate = int(math.ceil(2560 / len(self.ligand_feature_list_negative)))

            self.ligand_feature_list_negative = self.ligand_feature_list_negative * expand_rate
            self.ligand_structure_list_negative = self.ligand_structure_list_negative * expand_rate
            
            seed = random.randint(1,1000)
            for i in range(int(len(self.ligand_structure_list_negative)/expand_rate),len(self.ligand_structure_list_negative)):              
                torch.manual_seed(seed)
                self.ligand_feature_list_negative[i] = self.ligand_feature_list_negative[i][torch.randperm(self.ligand_feature_list_negative[i].size(0))]
                torch.manual_seed(seed)
                self.ligand_structure_list_negative[i] = self.ligand_structure_list_negative[i][torch.randperm(self.ligand_structure_list_negative[i].size(0))]
                torch.manual_seed(seed)
                self.ligand_structure_list_negative[i] = self.ligand_structure_list_negative[i][:,torch.randperm(self.ligand_structure_list_negative[i].size(1))]
                if (i%100 == 0):##Change the seed
                    seed = random.randint(1,1000)


             
        return 0

    def splitDataset(self):
        seed = random.randint(1,1000)
        random.seed(seed)
        random.shuffle(self.ligand_feature_list_positive)
        random.seed(seed)
        random.shuffle(self.ligand_structure_list_positive)
        seed = random.randint(1,1000)
        random.seed(seed)
        random.shuffle(self.ligand_feature_list_negative)
        random.seed(seed)
        random.shuffle(self.ligand_structure_list_negative)

        self.ligand_feature_list_all = self.ligand_feature_list_positive + self.ligand_feature_list_negative
        self.ligand_structure_list_all = self.ligand_structure_list_positive + self.ligand_structure_list_negative
        self.ligand_label_all = torch.cat([torch.ones(len(self.ligand_feature_list_positive)),torch.zeros(len(self.ligand_feature_list_negative))],dim=0)


        splitProp = 10

        self.ligand_feature_val_list_positive = self.ligand_feature_list_positive[0:int(len(self.ligand_feature_list_positive)/splitProp)]
        del self.ligand_feature_list_positive[0:int(len(self.ligand_feature_list_positive)/splitProp)]
        self.ligand_structure_val_list_positive = self.ligand_structure_list_positive[0:int(len(self.ligand_structure_list_positive)/splitProp)]
        del self.ligand_structure_list_positive[0:int(len(self.ligand_structure_list_positive)/splitProp)]

        self.ligand_feature_val_list_negative = self.ligand_feature_list_negative[0:int(len(self.ligand_feature_list_negative)/splitProp)]
        del self.ligand_feature_list_negative[0:int(len(self.ligand_feature_list_negative)/splitProp)]
        self.ligand_structure_val_list_negative = self.ligand_structure_list_negative[0:int(len(self.ligand_structure_list_negative)/splitProp)]
        del self.ligand_structure_list_negative[0:int(len(self.ligand_structure_list_negative)/splitProp)]


        self.ligand_feature_test_list_positive = self.ligand_feature_list_positive[0:int(len(self.ligand_feature_list_positive)/splitProp)]
        del self.ligand_feature_list_positive[0:int(len(self.ligand_feature_list_positive)/splitProp)]
        self.ligand_structure_test_list_positive = self.ligand_structure_list_positive[0:int(len(self.ligand_structure_list_positive)/splitProp)]
        del self.ligand_structure_list_positive[0:int(len(self.ligand_structure_list_positive)/splitProp)]

        self.ligand_feature_test_list_negative = self.ligand_feature_list_negative[0:int(len(self.ligand_feature_list_negative)/splitProp)]
        del self.ligand_feature_list_negative[0:int(len(self.ligand_feature_list_negative)/splitProp)]
        self.ligand_structure_test_list_negative = self.ligand_structure_list_negative[0:int(len(self.ligand_structure_list_negative)/splitProp)]
        del self.ligand_structure_list_negative[0:int(len(self.ligand_structure_list_negative)/splitProp)]




        print()
        if (len(self.ligand_feature_list_positive) == len(self.ligand_structure_list_positive)\
            and\
            len(self.ligand_feature_val_list_positive) == len(self.ligand_structure_val_list_positive)\
            and\
            len(self.ligand_feature_test_list_positive) == len(self.ligand_structure_test_list_positive))\
        and\
        (len(self.ligand_feature_list_negative) == len(self.ligand_structure_list_negative)\
         and\
         len(self.ligand_feature_val_list_negative) == len(self.ligand_structure_val_list_negative)\
         and\
         len(self.ligand_feature_test_list_negative) == len(self.ligand_structure_test_list_negative)):
            
            print("Training dataset: Positive(" + str(len(self.ligand_feature_list_positive))+ ") " +"Negative(" + str(len(self.ligand_feature_list_negative))+ ")")
            print("Validation dataset: Positive(" + str(len(self.ligand_feature_val_list_positive))+ ") " +"Negative(" + str(len(self.ligand_feature_val_list_negative))+ ")")
            print("Test dataset: Positive(" + str(len(self.ligand_feature_test_list_positive))+ ") " +"Negative(" + str(len(self.ligand_feature_test_list_negative))+ ")")
        else:
            print("Size error.")

        self.ligand_feature_list_val_all = self.ligand_feature_val_list_positive + self.ligand_feature_val_list_negative
        self.ligand_structure_list_val_all = self.ligand_structure_val_list_positive + self.ligand_structure_val_list_negative
        self.ligand_label_val_all = torch.cat([torch.ones(len(self.ligand_feature_val_list_positive)),torch.zeros(len(self.ligand_feature_val_list_negative))],dim=0)

        return 0

        

    def DataLoader(self, batch_size, turn_size):

        prop = self.prop

        
        samples_p = len(self.ligand_feature_list_positive)
        samples_n = len(self.ligand_feature_list_negative)
        
        indice_p = list(range(samples_p))
        indice_n = list(range(samples_n))

        random.shuffle(indice_p)
        random.shuffle(indice_n)
        indice_p = indice_p[0:int(turn_size)]
        indice_n = indice_n[0:int(turn_size*prop)]

        
        
        for i in range(0, turn_size, batch_size):
            seed = random.randint(1,1000)
            feature_set = []
            structure_set = []
            label_set = []
            for j in indice_p[i:min(i+batch_size,turn_size)]:
                feature_set.append(self.ligand_feature_list_positive[j])
                structure_set.append(self.ligand_structure_list_positive[j])
                label_set.append(1)

            for k in indice_n[i*prop:min((i+batch_size)*prop,turn_size*prop)]:
                feature_set.append(self.ligand_feature_list_negative[k])
                structure_set.append(self.ligand_structure_list_negative[k])
                label_set.append(0)
                
            random.seed(seed)    
            random.shuffle(feature_set)
            random.seed(seed)    
            random.shuffle(structure_set)
            random.seed(seed)    
            random.shuffle(label_set)
            label_set = torch.LongTensor(label_set)
            yield feature_set,structure_set,label_set


    def DataLoader_val(self, batch_size, turn_size):
        samples_p = len(self.ligand_feature_val_list_positive)
        samples_n = len(self.ligand_feature_val_list_negative)
        
        indice_p = list(range(samples_p))
        indice_n = list(range(samples_n))

        random.shuffle(indice_p)
        random.shuffle(indice_n)
        indice_p = indice_p[0:int(turn_size)]
        indice_n = indice_n[0:int(turn_size)]

        
        for i in range(0, turn_size, batch_size):
            seed = random.randint(1,1000)
            feature_set = []
            structure_set = []
            label_set = []
            for j,k in zip(indice_p[i:min(i+batch_size,turn_size)] , indice_n[i:min(i+batch_size,turn_size)]):
                feature_set.append(self.ligand_feature_val_list_positive[j])
                structure_set.append(self.ligand_structure_val_list_positive[j])
                label_set.append(1)
                feature_set.append(self.ligand_feature_val_list_negative[k])
                structure_set.append(self.ligand_structure_val_list_negative[k])
                label_set.append(0)
                
            random.seed(seed)    
            random.shuffle(feature_set)
            random.seed(seed)    
            random.shuffle(structure_set)
            random.seed(seed)    
            random.shuffle(label_set)
            label_set = torch.LongTensor(label_set)
            yield feature_set,structure_set,label_set




    def DataLoader_test(self, batch_size):
        for i in range(0,len(self.ligand_structure_test_list_positive),batch_size):
            yield self.ligand_feature_test_list_positive[i:i+batch_size]+self.ligand_feature_test_list_negative[i:i+batch_size], self.ligand_structure_test_list_positive[i:i+batch_size]+self.ligand_structure_test_list_negative[i:i+batch_size], torch.LongTensor([1]*batch_size+[0]*batch_size)


    def DataLoader_val_all(self, batch_size):
        for i in range(0,len(self.ligand_feature_list_val_all),batch_size):
            yield self.ligand_feature_list_val_all[i:i+batch_size],self.ligand_structure_list_val_all[i:i+batch_size],self.ligand_label_val_all[i:i+batch_size]

    def DataLoader_confuse(self, batch_size,c_p = 0.5):
        confuse_prop = c_p

        confuse_num = int(len(self.ligand_feature_val_list_positive)*confuse_prop)
        
        seed = random.randint(1,1000)
        random.seed(seed)
        random.shuffle(self.ligand_feature_val_list_positive)
        random.seed(seed)
        random.shuffle(self.ligand_structure_val_list_positive)
        seed = random.randint(1,1000)
        random.seed(seed)
        random.shuffle(self.ligand_feature_val_list_negative)
        random.seed(seed)
        random.shuffle(self.ligand_structure_val_list_negative)

        ligand_feature_list_conf = self.ligand_feature_val_list_positive + self.ligand_feature_val_list_negative
        ligand_structure_list_conf = self.ligand_structure_val_list_positive + self.ligand_structure_val_list_negative

        ligand_label_conf = torch.cat([torch.zeros(confuse_num),
                                       torch.ones(len(self.ligand_feature_val_list_positive)-confuse_num),
                                       torch.ones(confuse_num),
                                       torch.zeros(len(self.ligand_feature_val_list_negative)-confuse_num)],
                                      dim=0)
        for i in range(0,len(ligand_feature_list_conf),batch_size):
            yield ligand_feature_list_conf[i:i+batch_size],ligand_structure_list_conf[i:i+batch_size],ligand_label_conf[i:i+batch_size]
            
    def DataLoader_svm(self,train_num, val_num):

       seed = random.randint(1,1000)
       random.seed(seed)
       random.shuffle(self.ligand_dataset_svm_positive)

       seed = random.randint(1,1000)
       random.seed(seed)
       random.shuffle(self.ligand_dataset_svm_negative)


       svm_train = self.ligand_dataset_svm_positive[0:train_num]+self.ligand_dataset_svm_negative[0:train_num]
       svm_val = self.ligand_dataset_svm_positive[train_num+1:train_num+val_num+1]+self.ligand_dataset_svm_negative[train_num+1:train_num+val_num+1]
       
       svm_train_label = [1]*train_num + [0]*train_num
       svm_val_label = [1]*val_num + [0] *val_num

       seed = random.randint(1,1000)
       random.seed(seed)
       random.shuffle(svm_train)
       random.seed(seed)
       random.shuffle(svm_train_label)

       seed = random.randint(1,1000)
       random.seed(seed)
       random.shuffle(svm_val)
       random.seed(seed)
       random.shuffle(svm_val_label)

       return svm_train,svm_val,svm_train_label,svm_val_label

    def DataLoader_fcnn(self,train_num,val_num):
        seed = random.randint(1,1000)
        random.seed(seed)
        random.shuffle(self.ligand_dataset_fcnn_positive)

        seed = random.randint(1,1000)
        random.seed(seed)
        random.shuffle(self.ligand_dataset_fcnn_negative)

        fcnn_train = self.ligand_dataset_fcnn_positive[0:train_num]+self.ligand_dataset_fcnn_negative[0:train_num]
        fcnn_val = self.ligand_dataset_fcnn_positive[train_num+1:train_num+val_num+1]+self.ligand_dataset_fcnn_negative[train_num+1:train_num+val_num+1]

        fcnn_train_label = [1]*train_num + [0]*train_num
        fcnn_val_label = [1]*val_num + [0] *val_num
        fcnn_train_label = torch.LongTensor(fcnn_train_label)
        fcnn_val_label = torch.LongTensor(fcnn_val_label)
        
        fcnn_train = torch.cat(fcnn_train,dim=0)
        fcnn_val = torch.cat(fcnn_val,dim=0)
        return fcnn_train,fcnn_val,fcnn_train_label,fcnn_val_label

         
    def labelIden(self,l): 
        if (l[0] == 'A'):
            return int(23)
        elif (l[0] == 'B'):
            return int(33)
        elif (l[0] == 'C'):
            if (l[1] == '.'):
                if (l[2] == '1'):
                    return int(4)
                elif (l[2] == '2'):
                    return int(5)
                elif (l[2] == '3'):
                    return int(6)
                elif (l[2] == 'a'):
                    return int(7)
                elif (l[2] == 'c'):
                    return int(8)
                else:
                    print("Label error")
                    return None
            elif (l[1] == 'a'):
                return int(32)
            elif (l[1] == 'l'):
                return int(30)
            else:
                print("Label error")
                return None
        elif (l[0] == 'F'):
            return int(21)
        elif (l[0] == 'H'):
            if (len(l) == 1):
                return int(0)
            elif (l[2] == 's'):
                return int(1)
            elif (l[2] == 't'):
                return int(2)
            else:
                print("Label error")
                return None
        elif (l[0] == 'I'):
            return int(34)
        elif (l[0] == 'K'):
            return int(31)
        elif (l[0] == 'L'):
            return int(3)
        elif (l[0] == 'N'):
            if (l[1] == '.'):
                if (l[2] == '1'):
                    return int(9)
                elif (l[2] == '2'):
                    return int(10)
                elif (l[2] == '3'):
                    return int(11)
                elif (l[2] == '4'):
                    return int(12)
                elif (l[2] == 'p'):
                    return int(15)
                elif (l[3] == 'm'):
                    return int(13)
                elif (l[3] == 'r'):
                    return int(14)
                else:
                    print("Label error")
                    return None
            elif (l[1] == 'a'):
                return int(22)
            else:
                print("Label error")
                return None
        elif (l[0] == 'O'):
            if (l[2] == '2'):
                return int(16)
            elif (l[2] == '3'):
                return int(17)
            elif (l[2] == 'c'):
                return int(18)
            elif (l[2] == 's'):
                return int(19)
            elif (l[2] == 't'):
                return int(20)
            else:
                print("Label error")
                return None
        elif (l[0] == 'P'):
            return int(25)
        elif (l[0] == 'S'):
            if (l[1] == '.'):
                if (l[2] == '2'):
                    return int(26)
                elif (l[2] == '3'):
                    return int(27)
                elif (l[2] == 'o' and len(l) == 3):
                    return int(28)
                elif (l[2] == 'o' and len(l) ==4):
                    return int(29)
                else:
                    print("Label error")
                    return None
            elif (l[1] == 'i'):
                return int(24)
            else:
                print("Label error")
                return None
        else:
            print("Label error")
            return None
        
        return 0


##        if (l[0] == 'A'):
##            return int(7)
##        elif (l[0] == 'B'):
##            return int(14)
##        elif (l[0] == 'C'):
##            if (l[1] == '.'):
##                return int(0)
##            elif (l[1] == 'a'):
##                return int(13)
##            elif (l[1] == 'l'):
##                return int(11)
##            else:
##                print("Label error")
##                return None
##        elif (l[0] == 'F'):
##            return int(5)
##        elif (l[0] == 'H'):
##            return int(1)
##        elif (l[0] == 'I'):
##            return int(15)
##        elif (l[0] == 'K'):
##            return int(12)
##        elif (l[0] == 'L'):
##            return int(3)
##        elif (l[0] == 'N'):
##            if (l[1] == '.'):
##                return int(4)
##            elif (l[1] == 'a'):
##                return int(6)
##            else:
##                print("Label error")
##                return None
##        elif (l[0] == 'O'):
##            return int(2)
##        elif (l[0] == 'P'):
##            return int(9)
##        elif (l[0] == 'S'):
##            if (l[1] == '.'):
##                return int(10)
##            elif (l[1] == 'i'):
##                return int(8)
##            else:
##                print("Label error")
##                return None
##        else:
##            print("Label error")
##            return None
##        
##        return 0
##        
        
    def strSplit(self,s):
        s = s.replace("\n","")
        s = s.replace("\t"," ")
        s = s.split(" ")
        while '' in s:
            s.remove('')
        return s






        

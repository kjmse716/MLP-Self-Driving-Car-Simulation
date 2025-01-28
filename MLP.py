import numpy as np
import os
import random
np.random.seed()

class MLP:
    layer =2
    neurons_per_layer = 3
    input_dimension = 9
    learning_rate = 0.1
    epoch = 100
    ws = []
    
    def __init__(self,layer = 3,neurons_per_layer =9,learning_rate = 0.1,input_dimension = 4,epoch = 100):
        self.layer =layer
        self.neurons_per_layer = neurons_per_layer
        self.input_dimension = input_dimension
        self.learning_rate = learning_rate
        self.epoch = epoch
        self.ws = []

    def init_ws(self,test=False):
        if test == True:
            self.ws = [[[[-1.2],
                [1],
                [1]],[[0.3],
                        [1],
                        [1]]],
                [[[0.5],
                [0.4],
                [0.8]]]]
        else:
            first_layer = True
            for layer in range(self.layer):
                w=[]
                #如果是最後一層則只產生一組鍵結值
                for i in range(self.neurons_per_layer if layer != self.layer-1 else 1):
                    w.append(np.random.uniform(-0.01,0.01,size=(self.input_dimension+1 if first_layer==1 else self.neurons_per_layer+1,1)))#如果是第一層則產生input_dimension+1 size的鍵結值矩陣
                self.ws.append(w)
                first_layer=False 

    
    def neuron_forward(self,input,layer,index):
        input_temp = input.copy()
        input_temp.insert(0,-1)
        #print(self.ws[layer][index])
        #print(f"L={layer}I={index}")
        v = np.dot(input_temp,self.ws[layer][index])
        #print(v)
        y = 1 / (1 + np.exp(-v))
        #print(y)
        return y

        
    def neuron_backpropagation(self,ylist,ans,input_data):
        deltas= []
        for layer_index in range(self.layer-1,-1,-1):
            delta = []
            for n_index in range(self.neurons_per_layer-1,-1,-1)if layer_index!=self.layer-1 else range(1):
                yi = input_data if layer_index -1<0 else ylist[layer_index-1]
                yi = yi.copy()
                yi.insert(0,-1)
                o = ylist[layer_index][n_index]
                d=0
                if layer_index == self.layer-1:
                    d = ((ans-o)*o*(1-o))
                else:
                    delta_weight_sum=0
                    for i,dt in enumerate(deltas[0]):
                        delta_weight_sum += dt * self.ws[layer_index+1][-1-i][n_index][0]

                    d = o*(1-o)*(delta_weight_sum)
                #print("yi:",yi)
                w_alt = [[self.learning_rate * d *x for x in yi]]
                #print("w:",self.ws[layer_index][n_index])
                #print("w_alt:",w_alt)
                self.ws[layer_index][n_index] += np.transpose(w_alt)
                #print("new_w:",self.ws[layer_index][n_index])
                
                    
 
                delta.append(d)
            deltas.insert(0,delta)

    
    def train(self,data_path):
        data_path = os.path.join(data_path)
        datas = []
        ans = []
        with open(data_path,"r") as file:
            for line in file.readlines():
                val = line.strip().split(" ")
                val = list(map(lambda x:float(x),val))
                datas.append(val[:-1])
                
                ans.append(val[-1])
            #print(ans)
            #print(datas)
        
        ans = list(map(lambda x:(x + 40) / 80,ans))
        #print(ans)
        self.input_dimension = len(datas[0])
        self.init_ws() 
        #print(len(datas))
        for e in range(self.epoch):
            
            for i,data in enumerate(datas):
                #print(i)
                #print(data)
                self.calculation(data,True,ans[i]) 
            
            
        return self.ws

            
    def predict(self,input):
        self.input_dimension = len(input)
        result = self.calculation(input)
        print("nr=",result[-1][0])
        return (result[-1][0] * 80)-40
    
    
    def calculation(self,input,training = False,ans =""):
        y_list=[]
        first_layer =True
        for layer_index in range(self.layer):
            lr=[]
            for n_index in range(self.neurons_per_layer if layer_index!=self.layer-1 else 1):
                nr = self.neuron_forward(input if first_layer else y_list[-1],layer_index,n_index)
                lr.append(nr[0])
            y_list.append(lr)
            #print(lr)
            first_layer=False
            
        if training:
            self.neuron_backpropagation(y_list,ans,input)
        return y_list
        
        
        
if __name__ =="__main__":
    Test = MLP(layer=3, neurons_per_layer=9, input_dimension=3, learning_rate=0.5)
    data_path = r"E:\Lab\ANN\HW2\Documents\作業二\train4dAll.txt"
    #print(Test.neuron_forward([0.5,4,2,3],0,0))
    #print(Test.predict([0.5,4,2,3]))
    #print(Test.ws)

    
    Test.train(data_path)
    print(Test.predict([11.0396 ,05.8701 ,15.11]))


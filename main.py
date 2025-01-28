import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import MLP
from simple_playground import *


class plot:

    car_positions =[]
    def __init__(self,poss,state_logs):
        self.car_positions = poss
        self.plot(poss,state_logs)
        pass
    
    
    def read_coordinates(self,file_path):
        with open(file_path, 'r') as file:
            lines = file.readlines()
        
        start = tuple(map(float, lines[0].strip().split(',')))
        end_top_left = tuple(map(float, lines[1].strip().split(',')))
        end_bottom_right = tuple(map(float, lines[2].strip().split(',')))
        boundary_points = [tuple(map(float, line.strip().split(','))) for line in lines[3:]]
        
        return start, end_top_left, end_bottom_right, boundary_points
    
    
    def plot_track(self,start, end_top_left, end_bottom_right, boundary_points, car_positions,state_logs):
        #起點與終點
        ax.plot(start[0], start[1], 'go', label='Start (0,0)')
        
        rect = plt.Rectangle(end_top_left, end_bottom_right[0] - end_top_left[0], end_bottom_right[1] - end_top_left[1], linewidth=1, edgecolor='r', facecolor='none', label='End Area')
        ax.add_patch(rect)
        
        #邊界
        boundary_points.append(boundary_points[0]) 
        boundary_x, boundary_y = zip(*boundary_points)
        ax.plot(boundary_x, boundary_y, 'b-', label='Track Boundary')
        
        ax.plot([-6, 6], [0, 0], 'k--', label='Start Line')
        
        # 自走車與軌跡線段
        car_line, = ax.plot([], [], 'ro-', label='Car Position',markersize=np.pi*3**2)
        path_line, = ax.plot([], [], 'b', label='path')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.legend(loc='upper left')
        ax.set_aspect('equal', 'box')
        plt.title('Track Plot')
        plt.grid(True)
        def init():
            car_line.set_data([], [])
            path_line.set_data([],[])
            return car_line,

        def update(frame):
            if frame < len(car_positions):
                #car_x, car_y = zip(*car_positions[:frame+1])
                car_x, car_y = car_positions[frame]
                path_x, path_y = zip(*car_positions[:frame+1])
                
                car_line.set_data([car_x], [car_y])
                path_line.set_data(path_x,path_y)
                FD.set(round(state_logs[frame][0],3))
                RD.set(round(state_logs[frame][1],3))
                LD.set(round(state_logs[frame][2],3))
            return car_line,path_line,
        ani = animation.FuncAnimation(fig, update, frames=len(car_positions), init_func=init, blit=True, repeat=False)
        #plt.show()
        canvas.draw()
        #self.root.mainloop()
  
    def update_car_position(self,car_positions, new_position):
        car_positions.append(new_position)

        return car_positions


    def plot(self,car_positions,state_logs):
        # 開始繪圖
        ax.cla()
        file_path = '軌道座標點.txt'  
        start, end_top_left, end_bottom_right, boundary_points = self.read_coordinates(file_path)
        
        self.plot_track(start, end_top_left, end_bottom_right, boundary_points, car_positions,state_logs)
        time.sleep(0.3)  
    




from tkinter import filedialog
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
                                               NavigationToolbar2Tk)
from matplotlib.patches import Patch
    
root = tk.Tk()
root.title("自走車模擬")

LR= tk.DoubleVar(value=0.05)
EP= tk.IntVar(value=10)
LAYER = tk.IntVar(value=2)
NR = tk.IntVar(value=9)
FP = tk.StringVar(value="./input/train4dAll.txt")

FD = tk.DoubleVar()
RD = tk.DoubleVar()
LD = tk.DoubleVar()



#round(num, 2)

#W = tk.StringVar(value=f"w:\n\n\n")
#TrainingAC = tk.StringVar(value="epoch:0\nAC/epoch :0%")
#TestingAC = tk.StringVar(value="\nTesting AC :0%")
learningRate= LR.get()
epoch= EP.get()
#Acuracy = AC.get()

# 圖表初始化
fig, ax = plt.subplots()
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().grid(row=0,column=0,columnspan=2)


def check_Dimension(filepath):
    global D6D
    D6D=False
    with open(filepath,"r") as file:
        line = file.readline()
        if len(line.split(" "))==6:
            D6D=True
    print(f"D6D={D6D}")


#選擇檔案按鈕
def select():
    FP.set(filedialog.askopenfilename()) 
    print("reading data from:",FP.get())
    check_Dimension(FP.get())
    
    
#訓練按鈕
def training():
    global MLP1
    filepath = FP.get()
    check_Dimension(filepath)
    MLP1 = MLP.MLP(learning_rate=LR.get(),epoch=EP.get(),layer=LAYER.get(),neurons_per_layer=NR.get())
    
    MLP1.train(filepath)
    
    
    

#測試按鈕
def testing():
    global poss,state_logs,action_logs
    poss,state_logs,action_logs = run_example(MLP1,D6D)
    plot(poss,state_logs)
    canvas.draw()

def log_output():
    if D6D:
        with open("track6D.txt", 'w') as file:
            for i in range(len(poss)):
                line = f"{poss[i][0]} {poss[i][1]} {state_logs[i][0]} {state_logs[i][1]} {state_logs[i][2]} {action_logs[i]}\n"
                file.write(line)
    else:
        with open("track4D.txt", 'w') as file:
            for i in range(len(poss)):
                line = f"{state_logs[i][0]} {state_logs[i][1]} {state_logs[i][2]} {action_logs[i]}\n"
                file.write(line) 
        

#測試用函式
def train_f():
    global MLP1
    check_Dimension(FP.get)
    for epoch in range(0,5000,5):
        EP.set(epoch)

        MLP1 = MLP.MLP(learning_rate=LR.get(),epoch=EP.get(),layer=LAYER.get(),neurons_per_layer=NR.get())
        filepath = FP.get()
        MLP1.train(filepath)
    
        poss,state_logs = run_example(MLP1)

        if poss[-1][1]>=34:
            plot(poss,state_logs)
            print(f"epoch={epoch}")
            break



sensors_area = tk.LabelFrame(root,text= "sensors")
sensors_area.grid(row=1,column=0,columnspan=2,sticky=tk.W+tk.E)
variable_area = tk.Frame(root)
variable_area.grid(row=2,column=0)
actions_area = tk.LabelFrame(root,text= "Actions")
actions_area.grid(row=2,column=1,sticky=tk.W+tk.E)
sensors_area.columnconfigure(0, weight=1)
sensors_area.columnconfigure(1, weight=1)
sensors_area.columnconfigure(2, weight=1)


tk.Label(sensors_area, text="Front distance:").grid(row=0, column=0, sticky=tk.W)
tk.Label(sensors_area, textvariable=FD).grid(row=1, column=0, sticky=tk.W)
tk.Label(sensors_area, text="right45 distance:").grid(row=0, column=1, sticky=tk.W)
tk.Label(sensors_area, textvariable=RD).grid(row=1, column=1, sticky=tk.W)
tk.Label(sensors_area, text="left45 distance:").grid(row=0, column=2, sticky=tk.W)
tk.Label(sensors_area, textvariable=LD).grid(row=1, column=2, sticky=tk.W)

tk.Label(variable_area, text="layer").grid(row=0,column=0)
tk.Entry(variable_area, textvariable=LAYER).grid(row=1,column=0)
tk.Label(variable_area, text="neurn per layer").grid(row=2,column=0)
tk.Entry(variable_area, textvariable=NR).grid(row=3,column=0)
tk.Label(variable_area, text="learningRate").grid(row=4,column=0)
tk.Entry(variable_area, textvariable=LR).grid(row=5,column=0)
tk.Label(variable_area, text="epoch").grid(row=6,column=0)
tk.Entry(variable_area, textvariable=EP).grid(row=7,column=0)





tk.Button(actions_area,text="選擇檔案",command=select).grid(row=0,column=0)
btn = tk.Button(actions_area, text='訓練/train',command=training).grid(row=1,column=0)     
btn2 = tk.Button(actions_area, text='測試/test',command=testing) .grid(row=2,column=0)    
#tk.Button(actions_area,text="訓練至通過",command=train_f).grid(row=3,column=0)
tk.Button(actions_area,text="輸出log",command=log_output).grid(row=4,column=0)






root.mainloop()
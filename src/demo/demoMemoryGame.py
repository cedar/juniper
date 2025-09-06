from PIL import Image, ImageDraw
from IPython.display import display, clear_output
import numpy as np
import matplotlib.pyplot as plt

def toFC(x,y):
    return (x+191,y+168)
def eyeRP():
    return (660,200)
def eyeToFC(pos):
    return (145+pos[1]*10,60+pos[0]*10)   

class demoMemoryGame():
    def __init__(self, architecture, perception_field):
        self.recordings = []
        self.lastpos = eyeRP()
        rel_path = "img/memory"
        self.background = Image.open("img/background.png")
        self.eye = Image.open("img/eye.png")
        self.m_empty = Image.open(rel_path + "/m_empty.png")
        self.architecture = architecture
        self.eye_field = perception_field
        self.positions= [(50,40), (150,40), (250,40), (350,40),
                        (50,120), (150,120), (250,120), (350,120),
                        (50,200), (150,200), (250,200), (350,200)]
        np.random.shuffle(self.positions)
        self.m_pairs = {}
        for i in range(6):
            m_temp = Image.open(rel_path + "/m_"+str(i+1)+".png")
            demo_input0 = architecture.get_elements()["in"+str(2*i)]
            demo_input1 = architecture.get_elements()["in"+str((2*i)+1)]
            pos0 = self.positions[(2*i)]
            pos1 = self.positions[(2*i)+1]
            demo_input0._params["center"] = (3+pos0[1]//10, 3+pos0[0]//10, i)
            demo_input1._params["center"] = (3+pos1[1]//10, 3+pos1[0]//10, i)
            self.m_pairs["m_"+str(i+1)+"_0"] = { "image": m_temp, "visible": True, "solved": False, "demo_input" : demo_input0 , "position": pos0 }
            self.m_pairs["m_"+str(i+1)+"_1"] = { "image": m_temp, "visible": True, "solved": False, "demo_input" : demo_input1 , "position": pos1 }

    def find_element_at_point(self, point, box_size=10):
        x, y = point
        half_size = box_size // 2
        for key, data in self.m_pairs.items():
            cx, cy, cz = data["demo_input"]._params["center"] 
            left = cx - half_size
            right = cx + half_size
            top = cy - half_size
            bottom = cy + half_size
            if left <= x <= right and top <= y <= bottom:
                return key
        return None

    def tick(self,t):
        hidt = 4000
        show = (t % 100 == 0)
        if t == hidt:
            self.lastpos = eyeRP()
            for key, pair in self.m_pairs.items(): 
                self.m_pairs[key]["visible"] = False
                self.m_pairs[key]["demo_input"]._params["amplitude"] = 0.0
        if t == (hidt+100):
            for key, pair in self.m_pairs.items(): 
                self.m_pairs[key]["demo_input"]._params["amplitude"] = 1.0
        
        recording, ms_per_tick, timing = self.architecture.run_simulation(self.architecture.tick, [self.eye_field,self.eye_field+".activation","Shape Scene Memory","Intention Find First","Intention Find Same","CoS Same","CoD Same"], 1, print_timing=False)
      

        if (np.max(recording[0][3] > 0.9) or np.max(recording[0][4] > 0.9)) and np.max(recording[0][0] > 0.9):
            peak_pos_eye = np.argmax(recording[0][0])
            peak_pos_eye = np.unravel_index(peak_pos_eye, recording[0][0].shape)
            key = self.find_element_at_point(peak_pos_eye)
            if key is not None:
                if not self.m_pairs[key]["solved"]:
                    if not self.m_pairs[key]["visible"]:
                        self.m_pairs[key]["visible"] = True
                        show = True
        
        if np.max(recording[0][5] > 0.9): #CoS
            for key, pair in self.m_pairs.items():
                if not pair["solved"]:
                    if pair["visible"]:
                        self.m_pairs[key]["visible"] = False
                        self.m_pairs[key]["solved"] = True
                        self.m_pairs[key]["demo_input"]._params["amplitude"] = 0.0
                        show = True
                        
        if np.max(recording[0][6] > 0.9): #CoD
            for key, pair in self.m_pairs.items():
                if not pair["solved"]:
                    if pair["visible"]:
                        self.m_pairs[key]["visible"] = False
                        show = True
        
        if show:
            #print(t)
            background_img = Image.new("RGBA", self.background.size, color="white")
            background_img.paste(self.background)
            img = Image.new("RGBA", (447,294), color=(0, 0, 0, 0))
            for key, pair in self.m_pairs.items():    
                if not pair["solved"]:
                    if not pair["visible"]:
                        img.paste(self.m_empty, pair["position"], self.m_empty)
                    else:
                        img.paste(pair["image"], pair["position"], pair["image"])
            background_img.paste(img, toFC(0,0), img)
            if np.max(recording[0][0] > 0.9):
                peak_pos_eye = np.argmax(recording[0][0])
                peak_pos_eye = np.unravel_index(peak_pos_eye, recording[0][0].shape)
                self.lastpos = eyeToFC(peak_pos_eye)
                background_img.paste(self.eye, self.lastpos, self.eye) 
            else:
                #self.lastpos = eyeRP()
                background_img.paste(self.eye, self.lastpos, self.eye)
                
            cmap = 'jet'


            
            fig, axes = plt.subplots(2, 3, constrained_layout = True)
            axes[0][0].imshow(recording[0][2][:,:,0], vmin = 0, vmax = 1, cmap=cmap)   
            axes[0][1].imshow(recording[0][2][:,:,1], vmin = 0, vmax = 1, cmap=cmap)   
            axes[0][2].imshow(recording[0][2][:,:,2], vmin = 0, vmax = 1, cmap=cmap)
            axes[1][0].imshow(recording[0][2][:,:,3], vmin = 0, vmax = 1, cmap=cmap)   
            axes[1][1].imshow(recording[0][2][:,:,4], vmin = 0, vmax = 1, cmap=cmap)   
            axes[1][2].imshow(recording[0][2][:,:,5], vmin = 0, vmax = 1, cmap=cmap)

            plt.subplots_adjust(left=0.1, right=0.11, 
                    top=0.11, bottom=0.1, 
                    wspace=0.1, hspace=0.11)
            
            for n in range(2):
                for m in range(3):
                    axes[n][m].set_xticks([])
                    axes[n][m].set_yticks([])
            

            
            clear_output(wait=True)
            display(background_img)
            
            plt.show()
            plt.close(fig)

        return recording

    def run(self, num_ticks):  
        self.recordings = []
        self.architecture.reset_steps()
        for t in range(num_ticks):  
            self.recordings.append(self.tick(t))
        return self.recordings

from PIL import Image, ImageDraw
from IPython.display import display, clear_output
import numpy as np
import matplotlib.pyplot as plt

def toFC(x,y):
    return (x+191,y+168)
def eyeRP():
    return (660,200)
def eyeToFC(pos):
    return (145+pos[1],60+pos[0])   
def handRP():
    return (300,480)
def handToFC(pos):
    return (70+pos[1],160+pos[0])
def imgToFC(pos):
    dx = 338-240
    dy = 160-40
    return (pos[1]+dy,pos[0]+dx)

class demo2():
    def __init__(self, architecture, eye_field, hand_field):
        self.recordings = []
        rel_path = "img"
        self.background = Image.open(rel_path + "/background.png")
        self.eye = Image.open(rel_path + "/eye.png")
        self.hand = Image.open(rel_path + "/hand.png")
        self.light_off = Image.open(rel_path + "/light_off.png")
        self.light_on = Image.open(rel_path + "/light_on.png")

        
        self.light_positions= [(0,40), (120,40), (240,40)]
        np.random.shuffle(self.light_positions)

        self.architecture = architecture
        self.hand_field = hand_field
        self.eye_field = eye_field

        self.demo_input_0 = self.architecture.get_elements()["in0"]
        self.demo_input_1 = self.architecture.get_elements()["in1"]
        self.demo_input_2 = self.architecture.get_elements()["in2"]

    def tick(self,t):
        background_img = Image.new("RGBA", self.background.size, color="white")
        background_img.paste(self.background)
        img = Image.new("RGBA", (447,294), color=(0, 0, 0, 0))
        
        if t < 20:
            img.paste(self.light_off, self.light_positions[0], self.light_off)
            img.paste(self.light_off, self.light_positions[1], self.light_off)
            img.paste(self.light_off, self.light_positions[2], self.light_off)
            self.demo_input_0._params["center"] = imgToFC(self.light_positions[0])
            self.demo_input_0._params["amplitude"] = 2
            self.demo_input_1._params["center"] = imgToFC(self.light_positions[1])
            self.demo_input_1._params["amplitude"] = 2
            self.demo_input_2._params["center"] = imgToFC(self.light_positions[2])
            self.demo_input_2._params["amplitude"] = 2
        elif t >= 20:
            img.paste(self.light_on, self.light_positions[0], self.light_on)
            img.paste(self.light_on, self.light_positions[1], self.light_on)
            img.paste(self.light_on, self.light_positions[2], self.light_on)
            self.demo_input_0._params["center"] = imgToFC(self.light_positions[0])
            self.demo_input_0._params["amplitude"] = 5.
            self.demo_input_1._params["center"] = imgToFC(self.light_positions[1])
            self.demo_input_1._params["amplitude"] = 5.
            self.demo_input_2._params["center"] = imgToFC(self.light_positions[2])
            self.demo_input_2._params["amplitude"] = 5.
    
        recording, ms_per_tick, timing = self.architecture.run_simulation(self.architecture.tick, [self.eye_field,self.eye_field+".activation", self.hand_field, self.hand_field+".activation"], 1, print_timing=False)
    
        background_img.paste(img, toFC(0,0), img)
        if np.max(recording[0][0] > 0.99):
            peak_pos_eye = np.argmax(recording[0][0])
            peak_pos_eye = np.unravel_index(peak_pos_eye, recording[0][0].shape)
            peak_eye = Image.new("RGBA", (5,5), color="blue")
            img.paste(peak_eye, peak_pos_eye)
            background_img.paste(self.eye, eyeToFC(peak_pos_eye), self.eye) 
        else:
            background_img.paste(self.eye, eyeRP(), self.eye)
        
        if np.max(recording[0][2] > 0.99):
            peak_pos = np.argmax(recording[0][2])
            peak_pos = np.unravel_index(peak_pos, recording[0][2].shape)
            #print(peak_pos)
            peak = Image.new("RGBA", (5,5), color="red")
            img.paste(peak, peak_pos)
            background_img.paste(self.hand, handToFC(peak_pos), self.hand)
        else:
            background_img.paste(self.hand, handRP(), self.hand)
            
        clear_output(wait=True)
        display(background_img)

        return recording

    def run(self, num_ticks):  
        self.recordings = []
        self.architecture.reset_steps()
        for t in range(num_ticks):  
            self.recordings.append(self.tick(t))
        return self.recordings

    def plot(self):
        cmap = 'jet'
        for t in range(len(self.recordings)):
            fig, axes = plt.subplots(2, 2)
            
            axes[0][0].set_ylabel("Attention Field", rotation=0, size='large', labelpad = 100)
            axes[0][0].set_title('Activation')
            axes[0][0].imshow(self.recordings[t][0][1], vmin=-5, vmax=2, cmap=cmap)
            axes[0][1].imshow(self.recordings[t][0][0], vmin=0, vmax=1, cmap=cmap)
            axes[0][1].set_title('Sigm. Activation')
            
            axes[1][0].set_ylabel("Action Field", rotation=0, size='large', labelpad = 100)
            axes[1][0].imshow(self.recordings[t][0][3], vmin=-5, vmax=2, cmap=cmap)
            axes[1][1].imshow(self.recordings[t][0][2], vmin=0, vmax=1, cmap=cmap)
        
            axes[0][0].set_xticks([])
            axes[0][0].set_yticks([])
            axes[0][1].set_xticks([])
            axes[0][1].set_yticks([])
            axes[1][0].set_xticks([])
            axes[1][0].set_yticks([])
            axes[1][1].set_xticks([])
            axes[1][1].set_yticks([])
        
            clear_output(wait=True)
            plt.show()
            plt.close(fig)

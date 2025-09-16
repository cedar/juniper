from PIL import Image, ImageDraw, ImageFont
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

class demo3():
    def __init__(self, architecture, eye_field, hand_field):
        self.recordings = []
        rel_path = "demo/img"
        self.background = Image.open(rel_path + "/background.png")
        self.eye = Image.open(rel_path + "/eye.png")
        self.hand = Image.open(rel_path + "/hand.png")
        self.red_pill = Image.open(rel_path + "/red_pill.png")
        self.blue_pill = Image.open(rel_path + "/blue_pill.png")
        self.font = ImageFont.load_default(size=25)

        
        self.pill_positions = [(0,40), (240,40)]
        self.feature_red = (2,)
        self.feature_blue = (8,)

        self.architecture = architecture
        self.hand_field = hand_field
        self.eye_field = eye_field

        self.red_input = self.architecture.get_elements()["red_pill"]
        self.blue_input = self.architecture.get_elements()["blue_pill"]
        self.feature_cue = self.architecture.get_elements()["feature_cue"]

    def tick(self,t):
        background_img = Image.new("RGBA", self.background.size, color="white")
        background_img.paste(self.background)
        img = Image.new("RGBA", (447,294), color=(0, 0, 0, 0))
        draw = ImageDraw.Draw(background_img)
        
        img.paste(self.red_pill, self.pill_positions[0], self.red_pill)
        img.paste(self.blue_pill, self.pill_positions[1], self.blue_pill)
        self.red_input._params["center"] = self.feature_red+imgToFC(self.pill_positions[0])
        self.blue_input._params["center"] = self.feature_blue+imgToFC(self.pill_positions[1])

        if t < 20:
            self.feature_cue._params["amplitude"] = 0
        elif 20<=t<50:
            draw.text((385, 22), "RED", font=self.font, fill=(255, 0, 0, 255))
            self.feature_cue._params["amplitude"] = 2
        else:
            draw.text((375, 22), "BLUE", font=self.font, fill=(0, 0, 255, 255))
            self.feature_cue._params["center"] = (8,)
            self.feature_cue._params["amplitude"] = 2
            
    
        recording, ms_per_tick, timing = self.architecture.run_simulation(self.architecture.tick, [self.eye_field,self.eye_field+".activation", self.hand_field, self.hand_field+".activation", "Feature Map Field.activation"], 1, print_timing=False)
    
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
            fig, axes = plt.subplots(3, 2)
            


            axes[0][0].set_ylabel("Feature Map Field", rotation=0, size='large', labelpad = 100)
            #axes[0][0].set_title('Activation Space Proj.')
            im = axes[0][0].imshow(np.max(self.recordings[t][0][4], axis=0), vmin=-5, vmax=5, cmap=cmap)
            im = axes[0][1].plot(np.max(self.recordings[t][0][4], axis=(1,2)))
            axes[0][1].set_ylim(-5,5)
            #axes[0][1].set_title('Activation Feature Proj.')
            
            axes[1][0].set_ylabel("Attention Field", rotation=0, size='large', labelpad = 100)
            #axes[1][0].set_title('Activation')
            axes[1][0].imshow(self.recordings[t][0][1], vmin=-5, vmax=5, cmap=cmap)
            axes[1][1].imshow(self.recordings[t][0][0], vmin=0, vmax=1, cmap=cmap)
            #axes[1][1].set_title('Sigm. Activation')
            
            axes[2][0].set_ylabel("Action Field", rotation=0, size='large', labelpad = 100)
            axes[2][0].imshow(self.recordings[t][0][3], vmin=-5, vmax=5, cmap=cmap)
            axes[2][1].imshow(self.recordings[t][0][2], vmin=0, vmax=1, cmap=cmap)
        
            axes[0][0].set_xticks([])
            axes[0][0].set_yticks([])
            axes[0][1].set_xticks([])
            #axes[0][1].set_yticks([])
            axes[1][0].set_xticks([])
            axes[1][0].set_yticks([])
            axes[1][1].set_xticks([])
            axes[1][1].set_yticks([])
            axes[2][0].set_xticks([])
            axes[2][0].set_yticks([])
            axes[2][1].set_xticks([])
            axes[2][1].set_yticks([])
        
            clear_output(wait=True)
            plt.show()
            plt.close(fig)

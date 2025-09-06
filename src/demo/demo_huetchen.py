from PIL import Image, ImageDraw, ImageFont
from IPython.display import display, clear_output
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from src.demo.demo1 import demo1, toFC, eyeRP, eyeToFC, handRP, handToFC, imgToFC

def cupToBall(pos):
    return(pos[0] + 15, pos[1] + 70)
def ballToFC(pos):
    return (pos[1] + 15, pos[0] + 15)
def cupToFC(pos):
    return (pos[1] + 25, pos[0] + 70)
    

class demo_huetchen():
    def __init__(self, architecture, eye_field, hand_field):
        self.recordings = []
        rel_path = "img"
        self.background = Image.open(rel_path + "/background.png")
        self.eye = Image.open(rel_path + "/eye.png")
        self.hand = Image.open(rel_path + "/hand.png")
        self.cup_closed = Image.open(rel_path + "/cup_closed.png")
        self.cup_open = Image.open(rel_path + "/cup_open.png")
        self.ball = Image.open(rel_path + "/ball_trans.png")

        self.cup_positions = [(50,60), (170,60), (290,60)]
        np.random.shuffle(self.cup_positions)
        self.ball_position = cupToBall(self.cup_positions[-1])

        self.architecture = architecture
        self.hand_field = hand_field
        self.eye_field = eye_field

        self.demo_input = self.architecture.get_elements()["in0"]

        font_path = fm.findfont(fm.FontProperties(family="DejaVu Sans"))
        # print("Using font:", font_path)
        self.font = ImageFont.truetype(font_path, 50)

    def tick(self,t):
        background_img = Image.new("RGBA", self.background.size, color="white")
        background_img.paste(self.background)
        img = Image.new("RGBA", (447,294), color=(0, 0, 0, 0))

        # initial setup
        if t < 10:
            [img.paste(self.cup_closed, self.cup_positions[i], self.cup_closed) for i in range(3)]
        # open the cups and show the ball
        elif t < 40:
            [img.paste(self.cup_open, self.cup_positions[i], self.cup_open) for i in range(3)]
            img.paste(self.ball, self.ball_position, self.ball)
            self.demo_input._params["center"] = ballToFC(self.ball_position)
            self.demo_input._params["amplitude"] = 6
        # close the cup
        else:
            [img.paste(self.cup_closed, self.cup_positions[i], self.cup_closed) for i in range(3)]
            self.demo_input._params["amplitude"] = 0
    
        recording, ms_per_tick, timing = self.architecture.run_simulation(self.architecture.tick, [self.eye_field,self.eye_field+".activation", self.hand_field, self.hand_field+".activation","Memory Field","Memory Field.activation"], 1, print_timing=False)
    
        background_img.paste(img, toFC(0,0), img)
        if np.max(recording[0][0] > 0.5):
            peak_pos_eye = np.argmax(recording[0][0])
            peak_pos_eye = np.unravel_index(peak_pos_eye, recording[0][0].shape)
            peak_eye = Image.new("RGBA", (5,5), color="blue")
            img.paste(peak_eye, peak_pos_eye)
            background_img.paste(self.eye, eyeToFC(peak_pos_eye), self.eye) 
        else:
            background_img.paste(self.eye, eyeRP(), self.eye)
        
        if np.max(recording[0][2] > 0.5):
            peak_pos = np.argmax(recording[0][2])
            peak_pos = np.unravel_index(peak_pos, recording[0][2].shape)
            #print(peak_pos)
            peak = Image.new("RGBA", (5,5), color="red")
            img.paste(peak, peak_pos)
            background_img.paste(self.hand, handToFC(peak_pos), self.hand)
        else:
            background_img.paste(self.hand, handRP(), self.hand)

        if t >= 80:
            ImageDraw.Draw(background_img).text((360, 5), "Go", font=self.font, fill=(255, 255, 255))
            
            
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
            
            axes[0][0].set_ylabel("Attention Field", rotation=0, size='large', labelpad = 100)
            axes[0][0].set_title('Activation')
            axes[0][0].imshow(self.recordings[t][0][1], vmin=-5, vmax=5, cmap=cmap)
            axes[0][1].imshow(self.recordings[t][0][0], vmin=0, vmax=1, cmap=cmap)
            axes[0][1].set_title('Sigm. Activation')

            axes[1][0].set_ylabel("Memory Field", rotation=0, size='large', labelpad = 100)
            axes[1][0].imshow(self.recordings[t][0][5], vmin=-5, vmax=5, cmap=cmap)
            axes[1][1].imshow(self.recordings[t][0][4], vmin=0, vmax=1, cmap=cmap)
            
            axes[2][0].set_ylabel("Action Field", rotation=0, size='large', labelpad = 100)
            axes[2][0].imshow(self.recordings[t][0][3], vmin=-5, vmax=5, cmap=cmap)
            axes[2][1].imshow(self.recordings[t][0][2], vmin=0, vmax=1, cmap=cmap)
        
            axes[0][0].set_xticks([])
            axes[0][0].set_yticks([])
            axes[0][1].set_xticks([])
            axes[0][1].set_yticks([])
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

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
def FCToimg(pos):
    dx = 338-240
    dy = 160-40
    return (pos[1]-dy,pos[0]-dx)

class demo4():
    def __init__(self, architecture, eye_field, hand_field):
        self.recordings = []
        rel_path = "demo/img"
        self.background = Image.open(rel_path + "/background.png")
        self.eye = Image.open(rel_path + "/eye.png")
        self.hand = Image.open(rel_path + "/hand.png")
        self.red_pill = Image.open(rel_path + "/red_pill.png")
        self.blue_pill = Image.open(rel_path + "/blue_pill.png")
        self.white_pill = Image.open(rel_path + "/white_pill_rot.png")
        self.blue_clock = Image.open(rel_path + "/blue_clock.png")
        self.red_goggles = Image.open(rel_path + "/red_goggles_sm.png")
        self.green_clock = Image.open(rel_path + "/green_clock.png")
        self.font = ImageFont.load_default(size=25)

        self.scene_1_pos = [(147,82,), (172,325,)]
        self.scene_2_pos = [(82,147,), (207,221,), (77,350,)]

        self.architecture = architecture
        self.hand_field = hand_field
        self.eye_field = eye_field

        self.scene1_toggle = self.architecture.get_elements()["scene_1_gate"]
        self.scene2_toggle = self.architecture.get_elements()["scene_2_gate"]

        self.shape_cue = self.architecture.get_elements()["shape_cue"]
        self.color_cue = self.architecture.get_elements()["color_cue"]

        self.eye_pos = eyeRP()
        self.hand_pos = handRP()

    def tick(self,t):
        background_img = Image.new("RGBA", self.background.size, color="white")
        background_img.paste(self.background)
        img = Image.new("RGBA", (447,294), color=(0, 0, 0, 0))
        draw = ImageDraw.Draw(background_img)
        
        if t < 60:
            img.paste(self.red_pill, FCToimg(self.scene_1_pos[0]), self.red_pill)
            img.paste(self.blue_clock, FCToimg(self.scene_1_pos[1]), self.blue_clock)
            self.scene1_toggle._params["amplitude"] = 1
            self.scene2_toggle._params["amplitude"] = 0
        elif 60 <= t < 70:
            img.paste(self.blue_pill, FCToimg(self.scene_2_pos[0]), self.blue_pill)
            img.paste(self.green_clock, FCToimg(self.scene_2_pos[1]), self.green_clock)
            img.paste(self.red_goggles, FCToimg(self.scene_2_pos[2]), self.red_goggles)
            self.shape_cue._params["amplitude"] = 0
            self.color_cue._params["amplitude"] = 0
        elif 70 <= t < 120:
            background_img.paste(self.white_pill, (305, -60), self.white_pill)
            img.paste(self.blue_pill, FCToimg(self.scene_2_pos[0]), self.blue_pill)
            img.paste(self.green_clock, FCToimg(self.scene_2_pos[1]), self.green_clock)
            img.paste(self.red_goggles, FCToimg(self.scene_2_pos[2]), self.red_goggles)
            self.shape_cue._params["amplitude"] = 1
            self.color_cue._params["amplitude"] = 0
            self.scene1_toggle._params["amplitude"] = 0
            self.scene2_toggle._params["amplitude"] = 1
        elif 120 <= t < 130:
            img.paste(self.blue_pill, FCToimg(self.scene_2_pos[0]), self.blue_pill)
            img.paste(self.green_clock, FCToimg(self.scene_2_pos[1]), self.green_clock)
            img.paste(self.red_goggles, FCToimg(self.scene_2_pos[2]), self.red_goggles)
            self.shape_cue._params["amplitude"] = 0
            self.color_cue._params["amplitude"] = 0
        else:
            draw.text((385, 22), "BLUE", font=self.font, fill=(0, 0, 255, 255))
            img.paste(self.blue_pill, FCToimg(self.scene_2_pos[0]), self.blue_pill)
            img.paste(self.green_clock, FCToimg(self.scene_2_pos[1]), self.green_clock)
            img.paste(self.red_goggles, FCToimg(self.scene_2_pos[2]), self.red_goggles)
            self.shape_cue._params["amplitude"] = 0
            self.color_cue._params["amplitude"] = 1
            self.scene1_toggle._params["amplitude"] = 0
            self.scene2_toggle._params["amplitude"] = 1
    
        recording, ms_per_tick, timing = self.architecture.run_simulation(self.architecture.tick, [self.eye_field, self.eye_field+".activation", self.hand_field, self.hand_field+".activation","guidance_field","guidance_field.activation"], 1, print_timing=False)
        background_img.paste(img, toFC(0,0), img)
        if np.max(recording[0][0]) > 0.98:
            peak_pos_eye = np.argmax(recording[0][0])
            peak_pos_eye = np.unravel_index(peak_pos_eye, recording[0][0].shape)
            self.eye_pos = eyeToFC(peak_pos_eye)
            background_img.paste(self.eye, self.eye_pos, self.eye) 
        else:
            background_img.paste(self.eye, self.eye_pos, self.eye)
            
        if np.max(recording[0][2]) > 0.98:
            peak_pos = np.argmax(recording[0][2])
            peak_pos = np.unravel_index(peak_pos, recording[0][2].shape)
            self.hand_pos = handToFC(peak_pos)
            background_img.paste(self.hand, self.hand_pos, self.hand)
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
            
            axes[0][0].set_ylabel("Attention Field", rotation=0, size='large', labelpad = 100)
            axes[0][0].set_title('Activation')
            axes[0][1].set_title('Sigm. Activation')
            axes[1][0].set_ylabel("Action Field", rotation=0, size='large', labelpad = 100)
            axes[2][0].set_ylabel("Guidance Field", rotation=0, size='large', labelpad = 100)
            
            axes[0][0].imshow(self.recordings[t][0][0], vmin=0, vmax=1, cmap=cmap)
            im = axes[0][1].imshow(self.recordings[t][0][1], cmap=cmap)
            plt.colorbar(im, ax=axes[0][1])
            
            axes[1][0].imshow(self.recordings[t][0][2], vmin=0, vmax=1, cmap=cmap)
            im = axes[1][1].imshow(self.recordings[t][0][3], cmap=cmap)
            plt.colorbar(im, ax=axes[1][1])
            
            axes[2][0].imshow(self.recordings[t][0][4], vmin=0, vmax=1, cmap=cmap)
            im = axes[2][1].imshow(self.recordings[t][0][5], cmap=cmap)
            plt.colorbar(im, ax=axes[2][1])
        
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

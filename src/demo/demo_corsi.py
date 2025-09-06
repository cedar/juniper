from PIL import Image, ImageDraw, ImageFont
from IPython.display import display, clear_output
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from src.demo.demo1 import demo1, toFC, eyeRP, eyeToFC, handRP, handToFC, imgToFC


class demo_corsi_query(demo1):
    def __init__(self, architecture, eye_field, hand_field, use_query_inputs=True):
        super().__init__(architecture, eye_field, hand_field)
        self.demo_input = [self.architecture.get_elements()["in"+str(i)] for i in range(4)]
        if use_query_inputs:
            self.query_inputs = [self.architecture.get_elements()[f"query{i}"] for i in range(4)]
        else:
            self.query_inputs = None
        self.light_positions = [(-40,-30), (110,50), (270,-10), (240,110)]  # in FC coord
        self.num_objects = len(self.light_positions)
        np.random.shuffle(self.light_positions)
        self.font = ImageFont.truetype(fm.findfont(fm.FontProperties(family="DejaVu Sans")), 50)
        self.queried_object = None

    def tick(self, t):
        background_img = Image.new("RGBA", self.background.size, color="white")
        background_img.paste(self.background)
        img = Image.new("RGBA", (447,294), color=(0, 0, 0, 0))
        for i in range(self.num_objects):
            self.demo_input[i]._params["center"] = imgToFC(self.light_positions[i])
        
        # initial setup
        if t < 10:
            [img.paste(self.light_off, self.light_positions[i], self.light_off) for i in range(self.num_objects)]
        # item presentation n times sequentially
        t_present = 20
        t_pause = 15
        for n in range(self.num_objects):
            # present one item
            if (t >= 10 + t_present * n + t_pause * n) and (t < 10 + t_present * (n+1) + t_pause * n):
                [img.paste(self.light_off, self.light_positions[i], self.light_off) for i in range(self.num_objects)]
                img.paste(self.light_on, self.light_positions[n], self.light_on)
                self.demo_input[n]._params["amplitude"] = 10
            # pause period
            if (t >= 10 + t_present * (n+1) + t_pause * n) and (t < 10 + t_present * (n+1) + t_pause * (n+1)):
                [img.paste(self.light_off, self.light_positions[i], self.light_off) for i in range(self.num_objects)]
                self.demo_input[n]._params["amplitude"] = 0
        # short break before querying
        if (t >= 10 + t_present * self.num_objects + t_pause * self.num_objects) and (t < 20 + t_present * self.num_objects + t_pause * self.num_objects):
            [img.paste(self.light_off, self.light_positions[i], self.light_off) for i in range(self.num_objects)]
        # querying
        if t == 20 + t_present * self.num_objects + t_pause * self.num_objects:
            self.queried_object = np.random.choice(self.num_objects)
            self.query_inputs[self.queried_object]._params["amplitude"] = 3
        if t >= 20 + t_present * self.num_objects + t_pause * self.num_objects:
            [img.paste(self.light_off, self.light_positions[i], self.light_off) for i in range(self.num_objects)]
            ImageDraw.Draw(background_img).text((390, 8), str(self.queried_object), font=self.font, fill=(255, 255, 255))
            
        recording_list = [self.eye_field,self.eye_field+".activation", self.hand_field, self.hand_field+".activation"]
        recording_list += [f'map{i}.activation' for i in range(4)]
        recording_list += [f'cosfield{i}.activation' for i in range(4)]
        recording_list += [f'ord{i}.activation' for i in range(4)]
        recording_list += ['u.activation', 'v.activation']
        recording_list += ['int.activation', 'cos.activation']
        recording_list += [f'mem{i}.activation' for i in range(4)]
        recording_list += [f'readout{i}.activation' for i in range(4)]
        recording, ms_per_tick, timing = self.architecture.run_simulation(self.architecture.tick, recording_list, 1, print_timing=False)

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
            peak = Image.new("RGBA", (5,5), color="red")
            img.paste(peak, peak_pos)
            background_img.paste(self.hand, handToFC(peak_pos), self.hand)
        else:
            background_img.paste(self.hand, handRP(), self.hand)
            
        clear_output(wait=True)
        display(background_img)

        return recording

    
    def plot(self):
        def sum_recordings(init_ind):
            arr = np.zeros(4)
            for i in range(4):
                arr = arr + self.recordings[t][0][init_ind + i*2][0]
            return arr
        def scalar_to_vec(init_ind):
            arr = np.zeros(4)
            for i in range(4):
                summand = np.zeros(4)
                summand[i] = self.recordings[t][0][init_ind + i][0]
                arr = arr + summand
            return arr
        vmin_acti = -5
        vmax_acti = 5
        vmin_sig = 0
        vmax_sig = 1
        cmap = 'jet'

        for t in range(len(self.recordings)):
            fig, axes = plt.subplots(2, 4)
            for i in range(2):
                for j in range(4):
                    axes[i][j].set_xticks([])
                    axes[i][j].set_yticks([])
                    axes[i][j].set_box_aspect(294/447)
            fig.subplots_adjust(hspace=-0.6)
            
            axes[0][0].set_title("Attention Field", size='large')
            axes[0][0].imshow(self.recordings[t][0][1], vmin=vmin_acti, vmax=vmax_acti, cmap=cmap)

            axes[0][1].set_title("Ordinal nodes", size='large')
            axes[0][1].scatter(range(4), np.clip(scalar_to_vec(12), vmin_acti, vmax_acti))
            axes[0][1].margins(x=0.1)
            axes[0][1].set_ylim(vmin_acti-1, vmax_acti+1)
            axes[0][1].autoscale(enable=True, axis="x", tight=False)
            line = axes[0][1].axhline(0, color='grey', alpha=0.5)
            line.set_animated(False)
            line.set_clip_on(True)
            line.set_in_layout(False) 


            axes[0][2].set_title("Intention & CoS", size='large')
            axes[0][2].scatter(range(2), np.clip(scalar_to_vec(16)[2:], vmin_acti, vmax_acti))     
            axes[0][2].margins(x=0.4)
            axes[0][2].set_ylim(vmin_acti-1, vmax_acti+1)
            axes[0][2].autoscale(enable=True, axis="x", tight=False)
            line = axes[0][2].axhline(0, color='grey', alpha=0.5)
            line.set_animated(False)
            line.set_clip_on(True)
            line.set_in_layout(False) 
            
            axes[0][3].set_title("Action Field", size='large')
            axes[0][3].imshow(self.recordings[t][0][3], vmin=vmin_acti, vmax=vmax_acti, cmap=cmap)

            axes[1][0].set_ylabel("Mental map", rotation=0, size='large', labelpad = 50)
            [axes[1][i].imshow(self.recordings[t][0][i+4], vmin=vmin_acti, vmax=vmax_acti, cmap=cmap) for i in range(4)]
            
            clear_output(wait=True)
            plt.show()
            plt.close(fig)
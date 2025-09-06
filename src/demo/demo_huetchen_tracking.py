from PIL import Image, ImageDraw, ImageFont
from IPython.display import display, clear_output
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from src.demo.demo_huetchen import demo_huetchen, toFC, eyeRP, eyeToFC, handRP, handToFC, imgToFC, cupToBall, ballToFC, cupToFC


class demo_huetchen_tracking(demo_huetchen):
    def __init__(self, architecture, eye_field, hand_field):
        super().__init__(architecture, eye_field, hand_field)
        self.cup_inputs = [self.architecture.get_elements()["in"+str(i)] for i in range(1,4)] 
        self.cup_positions = np.array(self.cup_positions)

    def tick(self,t):
        background_img = Image.new("RGBA", self.background.size, color="white")
        background_img.paste(self.background)
        img = Image.new("RGBA", (447,294), color=(0, 0, 0, 0))

        # initial setup
        if t < 10:
            [img.paste(self.cup_closed, tuple(self.cup_positions[i]), self.cup_closed) for i in range(3)]
        # open the cups and show the ball
        elif t < 30:
            [img.paste(self.cup_open, tuple(self.cup_positions[i]), self.cup_open) for i in range(3)]
            img.paste(self.ball, self.ball_position, self.ball)
            self.demo_input._params["center"] = ballToFC(self.ball_position)
            self.demo_input._params["amplitude"] = 3
        # close the cups
        elif t < 40:
            [img.paste(self.cup_closed, tuple(self.cup_positions[i]), self.cup_closed) for i in range(3)]
            self.demo_input._params["amplitude"] = 0
        # move cups
        elif t < 75:
            dxdy = np.random.randint(-10, 11, (3,2))
            self.cup_positions = self.cup_positions + dxdy
            self.ball_position = (self.ball_position[0]+dxdy[2,0], self.ball_position[0]+dxdy[2,1])
            self.demo_input._params["center"] = ballToFC(self.ball_position)
            [img.paste(self.cup_closed, tuple(self.cup_positions[i]), self.cup_closed) for i in range(3)]
            # update weak cup inputs to allow tracking
            for i in range(3): 
                self.cup_inputs[i]._params["center"] = cupToFC(self.cup_positions[i])
        # add go signal
        elif t < 80:
            [img.paste(self.cup_closed, tuple(self.cup_positions[i]), self.cup_closed) for i in range(3)]
            ImageDraw.Draw(background_img).text((360, 5), "Go", font=self.font, fill=(255, 255, 255))
        else:
            [img.paste(self.cup_closed, tuple(self.cup_positions[i]), self.cup_closed) for i in range(3)]
            
    
        recording, ms_per_tick, timing = self.architecture.run_simulation(self.architecture.tick, [self.eye_field,self.eye_field+".activation", self.hand_field, self.hand_field+".activation"], 1, print_timing=False)

        if t > 10:
            ball_pos = ballToFC(self.ball_position)
            ImageDraw.Draw(img).rectangle([ball_pos[0], ball_pos[1], ball_pos[0]+10, ball_pos[1]+10], fill="blue")
            for i in range(3):
                cup_pos = cupToFC(self.cup_positions[i]) 
                ImageDraw.Draw(img).rectangle([cup_pos[0], cup_pos[1], cup_pos[0]+10, cup_pos[1]+10], fill="red")
    
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

        # if t >= 80:
        #     ImageDraw.Draw(background_img).text((360, 5), "Go", font=self.font, fill=(255, 255, 255))

            
            
        clear_output(wait=True)
        display(background_img)
            
        return recording
from juniper.FrameGraph import FrameGraph
from juniper.Transform import Transform
from juniper.steps.CoordinateTransformation import CoordinateTransformation
from juniper.steps.TCPSocket import TCPSocket
from juniper.steps.PinHoleBackProjector import PinHoleBackProjector
from juniper.steps.NeuralField import NeuralField
from juniper.steps.RangeImageVectorMapper import RangeImageVectorMapper

import jax
import jax.numpy as jnp
import jax.debug as jdbg
#-----------------------------------------------------------------------------------
v_base_to_cam = jnp.array([0, 0.01, 0.908, 1])
M_base_2_cam = jnp.eye(4)
M_base_2_cam = M_base_2_cam.at[:,3].set(v_base_to_cam)
def base_to_cam_joint(joint_angles):
    return M_base_2_cam

v_joint_to_cam = jnp.array([0,0,0.0,1]) #jnp.array([0,0,0.097,1])
def cam_base_to_joint(joint_angles):
    tilt = joint_angles[0][0]
    pan = joint_angles[1][0]

    #pan_mat = jnp.array([[1,0,0,0], [0, jnp.cos(pan), -jnp.sin(pan), 0], [0, jnp.sin(pan), jnp.cos(pan), 0], [0,0,0,1]])
    tilt_mat = jnp.array([[jnp.cos(tilt),0,jnp.sin(tilt),0], [0, 1, 0, 0], [-jnp.sin(tilt), 0, jnp.cos(tilt), 0], [0,0,0,1]])
    pan_mat = jnp.array([[jnp.cos(pan),-jnp.sin(pan),0,0], [jnp.sin(pan), jnp.cos(pan), 0, 0], [0, 0, 1, 0], [0,0,0,1]])
    Rot_mat = pan_mat @ tilt_mat
    Rot_mat = Rot_mat.T
    
    Rot_mat = Rot_mat.at[:,3].set(v_joint_to_cam)
    #Rot_mat = jnp.linalg.inv(Rot_mat)
    #jdbg.print('{}',Rot_mat)

    return Rot_mat


v_base_to_field = jnp.array([0.1, -0.99, 0.06, 1])
R_base_to_field = jnp.array([[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1]]) * 100
R_base_to_field = R_base_to_field.at[:,3].set(v_base_to_field)
def base_to_field(joint_angles):
    return R_base_to_field


# make 3 test transforms, one for arbitrary translations, and one for arbitrary pans, and one for arbitrary tilts. Once they work, i can start to combine them
#------------------------------------------------------------------------------------



def get_architecture(args):

    frame_graph = FrameGraph(params={})

    frame_graph.add_edge(source = "base", target = "cam_joint", Transform = Transform({"M_func": base_to_cam_joint}))
    frame_graph.add_edge(source = "cam_base", target = "cam_joint", Transform=Transform({"M_func": cam_base_to_joint}))
    frame_graph.add_edge(source = "base", target="field", Transform=Transform({"M_func": base_to_field}))

    #coord_transf1 = CoordinateTransformation("coord_transf1", {"FrameGraph": frame_graph, "source_frame": "cam_base", "target_frame": "cam_joint"})


    img_proj = PinHoleBackProjector("img_proj", {"img_shape":(299,448), "focal_length": 0.01, "frustrum_angles": (90,60)})
    rim1 = RangeImageVectorMapper("rim1", {"img_shape": (180,180), "pan_range":(-jnp.pi, jnp.pi), "tilt_range":(-jnp.pi/2,jnp.pi/2), "output_type": "image"})
    rim2 = RangeImageVectorMapper("rim2", {"img_shape": (180,180), "pan_range":(-jnp.pi, jnp.pi), "tilt_range":(-jnp.pi/2,jnp.pi/2), "output_type": "vector"})
    rim3 = RangeImageVectorMapper("rim3", {"img_shape": (180,180), "pan_range":(-jnp.pi, jnp.pi), "tilt_range":(-jnp.pi/2,jnp.pi/2), "output_type": "image"})


    tcp_reader = TCPSocket("tcp_reader", {"mode": "read", "ip": "127.0.0.1", "port": 50025, "shape": (299,448), 'time_step': 0.02})
    #tcp_reader_joints = TCPSocket("tcp_reader_joints", {"mode": "read", "ip": "127.0.0.1", "port": 50022, "shape": (2,1), 'time_step': 0.02})

    tcp_lidar_writer = TCPSocket("tcp_lidar_writer", {"mode": "write", "ip": "127.0.0.1", "port": 50001, "shape": (299,448), 'time_step': 0.02})
    tcp_head_writer = TCPSocket("tcp_head_writer", {"mode": "write", "ip": "127.0.0.1", "port": 50002, "shape": (180,180), 'time_step': 0.02})
    tcp_allo_writer = TCPSocket("tcp_writer", {"mode": "write", "ip": "127.0.0.1", "port": 50003, "shape": (180,180), 'time_step': 0.02})


    nf1 = NeuralField("nf1", {"shape": (180,180), "sigmoid": "AbsSigmoid", "beta": 100, "theta": 0, "resting_level": -5, "global_inhibition": 0, "input_noise_gain": 0, "tau": 0.1})
    nf2 = NeuralField("nf2", {"shape": (299,448), "sigmoid": "AbsSigmoid", "beta": 100, "theta": 0, "resting_level": -5, "global_inhibition": 0, "input_noise_gain": 0, "tau": 0.1})
    nf3 = NeuralField("nf3", {"shape": (180,180), "sigmoid": "AbsSigmoid", "beta": 100, "theta": 0, "resting_level": -5, "global_inhibition": 0, "input_noise_gain": 0, "tau": 0.1})



    tcp_reader >> tcp_lidar_writer >> nf2
    tcp_reader >> img_proj >> rim1 >> tcp_head_writer >> nf3

    rim1 >> rim2 >> rim3 >> tcp_allo_writer >> nf1
    #tcp_reader_joints >> "coord_transf1.in1" 
    


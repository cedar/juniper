from juniper.FrameGraph import FrameGraph
from juniper.Transform import Transform
from juniper.steps.CoordinateTransformation import CoordinateTransformation
from juniper.steps.TCPSocket import TCPSocket
from juniper.steps.PinHoleBackProjector import PinHoleBackProjector
from juniper.steps.NeuralField import NeuralField
from juniper.steps.RangeImageVectorMapper import RangeImageVectorMapper
from juniper.steps.FieldVectorMapper import FieldVectorMapper
from juniper.steps.CompressAxes import CompressAxes

import jax
import jax.numpy as jnp
import jax.debug as jdbg
#-----------------------------------------------------------------------------------
v_base_to_cam = jnp.array([0, 0.01, 0.908, 1])
M_base_2_cam = jnp.eye(4)
M_base_2_cam = M_base_2_cam.at[:,3].set(v_base_to_cam)
def base_to_cam_joint(joint_angles):
    return M_base_2_cam

v_joint_to_cam = jnp.array([0,0,0.097])
def joint_to_cam_base(joint_angles):
    tilt = joint_angles[1][0] / 180 * jnp.pi 
    pan =  -joint_angles[0][0] / 180 * jnp.pi 

    tilt_mat = jnp.array([[jnp.cos(tilt),0,jnp.sin(tilt)], [0, 1, 0], [-jnp.sin(tilt), 0, jnp.cos(tilt)]])
    pan_mat = jnp.array([[jnp.cos(pan),-jnp.sin(pan),0], [jnp.sin(pan), jnp.cos(pan), 0], [0, 0, 1]])
    Rot_mat = tilt_mat @ pan_mat
    Rot_mat = Rot_mat
    cam_offset = Rot_mat @ v_joint_to_cam

    M = jnp.eye(4)
    M = M.at[:3, :3].set(Rot_mat)
    M = M.at[:3, 3].set(cam_offset)

    return M

def cam_base_to_joint(joint_angles):
    T = joint_to_cam_base(joint_angles)
    T = jnp.linalg.inv(T)
    return T

v_base_to_field = jnp.array([0.1, -0.99, 0.06, 1])
R_base_to_field = jnp.array([[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1]])
R_base_to_field = R_base_to_field.at[:,3].set(v_base_to_field) * 100
def base_to_field(joint_angles):
    return R_base_to_field

x = 0.0
y = 0.0
z = 0.
M_x = jnp.array([[1,0,0,x], [0,1,0,y], [0,0,1,z], [0,0,0,1]])
def translation(joint_angles):
    return M_x

tilt = jnp.pi/4
pan = 0
M_r = jnp.array([[jnp.cos(pan), -jnp.sin(pan),0,0], [jnp.sin(pan), jnp.cos(pan),0,0], [0,0,1,0], [0,0,0,1]]) @ jnp.array([[jnp.cos(tilt),0,jnp.sin(tilt),0], [0,1,0,0], [-jnp.sin(tilt),0,jnp.cos(tilt),0], [0,0,0,1]])
def rotation(joint_angles):
    #jdbg.print('{}', joint_angles/180*jnp.pi)
    return M_r

# make 3 test transforms, one for arbitrary translations, and one for arbitrary pans, and one for arbitrary tilts. Once they work, i can start to combine them
#------------------------------------------------------------------------------------



def get_architecture(args):

    frame_graph = FrameGraph(params={})

    frame_graph.add_edge(source = "base", target = "cam_joint", transform = Transform({"M_func": base_to_cam_joint}))
    frame_graph.add_edge(source = "cam_joint", target = "cam", transform=Transform({"M_func": joint_to_cam_base}))
    frame_graph.add_edge(source = "base", target="field", transform=Transform({"M_func": base_to_field}))

    coord_transf1 = CoordinateTransformation("coord_transf1", {"FrameGraph": frame_graph, "source_frame": "cam", "target_frame": "field"})


    img_proj = PinHoleBackProjector("img_proj", {"img_shape":(299,448), "focal_length": 0.01, "frustrum_angles": (90,60)})
    rim1 = RangeImageVectorMapper("rim1", {"img_shape": (180,180), "pan_range":(-jnp.pi, jnp.pi), "tilt_range":(-jnp.pi/2,jnp.pi/2), "output_type": "image"})
    rim2 = RangeImageVectorMapper("rim2", {"img_shape": (180,180), "pan_range":(-jnp.pi, jnp.pi), "tilt_range":(-jnp.pi/2,jnp.pi/2), "output_type": "vector"})
    rim3 = RangeImageVectorMapper("rim3", {"img_shape": (180,180), "pan_range":(-jnp.pi, jnp.pi), "tilt_range":(-jnp.pi/2,jnp.pi/2), "output_type": "image"})

    fm1 = FieldVectorMapper("fm1", params={"field_shape": (60,180,20), "output_type": "field"})
    comp = CompressAxes("comp", params={"axis":(2,), "compression_type": "Maximum"})


    tcp_reader = TCPSocket("tcp_reader", {"mode": "read", "ip": "127.0.0.1", "port": 50025, "shape": (299,448), 'time_step': 0.02})
    tcp_reader_joints = TCPSocket("tcp_reader_joints", {"mode": "read", "ip": "127.0.0.1", "port": 50022, "shape": (2,1), 'time_step': 0.02})

    tcp_lidar_writer = TCPSocket("tcp_lidar_writer", {"mode": "write", "ip": "127.0.0.1", "port": 50001, "shape": (299,448), 'time_step': 0.02})
    tcp_head_writer = TCPSocket("tcp_head_writer", {"mode": "write", "ip": "127.0.0.1", "port": 50002, "shape": (180,180), 'time_step': 0.02})
    tcp_allo_writer1 = TCPSocket("tcp_writer1", {"mode": "write", "ip": "127.0.0.1", "port": 500123, "shape": (180,180), 'time_step': 0.02})
    tcp_allo_writer = TCPSocket("tcp_writer", {"mode": "write", "ip": "127.0.0.1", "port": 50003, "shape": (60,180), 'time_step': 0.02})


    nf1 = NeuralField("nf1", {"shape": (180,180), "sigmoid": "AbsSigmoid", "beta": 100, "theta": 0, "resting_level": -5, "global_inhibition": 0, "input_noise_gain": 0, "tau": 0.1})
    nf2 = NeuralField("nf2", {"shape": (299,448), "sigmoid": "AbsSigmoid", "beta": 100, "theta": 0, "resting_level": -5, "global_inhibition": 0, "input_noise_gain": 0, "tau": 0.1})
    nf3 = NeuralField("nf3", {"shape": (180,180), "sigmoid": "AbsSigmoid", "beta": 100, "theta": 0, "resting_level": -5, "global_inhibition": 0, "input_noise_gain": 0, "tau": 0.1})
    nf4 = NeuralField("nf4", {"shape": (60,180), "sigmoid": "AbsSigmoid", "beta": 100, "theta": 0, "resting_level": -5, "global_inhibition": 0, "input_noise_gain": 0, "tau": 0.1})



    tcp_reader >> tcp_lidar_writer >> nf2
    tcp_reader >> img_proj >> rim1 >> tcp_head_writer >> nf3

    img_proj >> coord_transf1 >> rim3 >> tcp_allo_writer1 >> nf1
    tcp_reader_joints >> "coord_transf1.in1" 

    coord_transf1 >> fm1 >> comp >> tcp_allo_writer >> nf4

    


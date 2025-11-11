from juniper.FrameGraph import FrameGraph
from juniper.Transform import Transform
from juniper.steps.CoordinateTransformation import CoordinateTransformation
from juniper.steps.TCPReader import TCPReader
from juniper.steps.TCPWriter import TCPWriter
from juniper.steps.PinHoleBackProjector import PinHoleBackProjector
from juniper.steps.NeuralField import NeuralField
from juniper.steps.FieldToVectors import FieldToVectors
from juniper.steps.VectorsToField import VectorsToField
from juniper.steps.VectorsToRangeImage import VectorsToRangeImage
from juniper.steps.RangeImageToVectors import RangeImageToVectors

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

v_joint_to_cam = jnp.array([0,0,0.097])*0
def joint_to_cam_base(joint_angles):
    tilt = joint_angles[1][0] / 180 * jnp.pi 
    pan =  -joint_angles[0][0] / 180 * jnp.pi 

    tilt_mat = jnp.array([[jnp.cos(tilt),0,jnp.sin(tilt)], [0, 1.0, 0], [-jnp.sin(tilt), 0, jnp.cos(tilt)]])
    pan_mat = jnp.array([[jnp.cos(pan),-jnp.sin(pan),0], [jnp.sin(pan), jnp.cos(pan), 0], [0, 0, 1]])
    Rot_mat = tilt_mat @ pan_mat
    Rot_mat = Rot_mat
    cam_offset = jnp.linalg.inv(Rot_mat) @ v_joint_to_cam

    M = jnp.eye(4)
    M = M.at[:3, :3].set(Rot_mat)
    M = M.at[:3, 3].set(cam_offset)

    return M

def cam_base_to_joint(joint_angles):
    T = joint_to_cam_base(joint_angles)
    T = jnp.linalg.inv(T)
    return T

R_base_to_field = jnp.array([[1.0,0,0,0], [0,-1,0,0], [0,0,-1,0], [0,0,0,1]])
jdbg.print('{}', R_base_to_field)
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

#------------------------------------------------------------------------------------



def get_architecture(args):

    frame_graph = FrameGraph(params={})

    frame_graph.add_edge(source = "base", target = "cam_joint", transform = Transform({"M_func": base_to_cam_joint}))
    frame_graph.add_edge(source = "cam_joint", target = "cam", transform=Transform({"M_func": joint_to_cam_base}))
    frame_graph.add_edge(source = "base", target="field", transform=Transform({"M_func": base_to_field}))

    coord_transf1 = CoordinateTransformation("coord_transf1", {"FrameGraph": frame_graph, "source_frame": "cam", "target_frame": "field"})
    coord_transf2 = CoordinateTransformation("coord_transf2", {"FrameGraph": frame_graph, "source_frame": "field", "target_frame": "cam"})
    coord_transf3 = CoordinateTransformation("coord_transf3", {"FrameGraph": frame_graph, "source_frame": "cam", "target_frame": "cam_joint"})
    coord_transf4 = CoordinateTransformation("coord_transf4", {"FrameGraph": frame_graph, "source_frame": "field", "target_frame": "cam"})


    img_proj = PinHoleBackProjector("img_proj", {"img_shape":(299,448), "focal_length": 0.01, "frustrum_angles": (90,60)})
    rim1 = VectorsToRangeImage("rim1", {"image_shape": (180,180), "pan_range":(-jnp.pi, jnp.pi), "tilt_range":(-jnp.pi/2,jnp.pi/2)})
    rim2 = VectorsToRangeImage("rim2", {"image_shape": (180,180), "pan_range":(-jnp.pi, jnp.pi), "tilt_range":(-jnp.pi/2,jnp.pi/2)})
    rim3 = VectorsToRangeImage("rim3", {"image_shape": (180,180), "pan_range":(-jnp.pi, jnp.pi), "tilt_range":(-jnp.pi/2,jnp.pi/2)})
    rim4 = VectorsToRangeImage("rim4", {"image_shape": (180,180), "pan_range":(-jnp.pi, jnp.pi), "tilt_range":(-jnp.pi/2,jnp.pi/2)})
    rim5 = VectorsToRangeImage("rim5", {"image_shape": (180,180), "pan_range":(-jnp.pi, jnp.pi), "tilt_range":(-jnp.pi/2,jnp.pi/2)})

    rim1_5 = RangeImageToVectors("rim1_5", {"image_shape": (180,180), "pan_range":(-jnp.pi, jnp.pi), "tilt_range":(-jnp.pi/2,jnp.pi/2)})
    rim1_6 = VectorsToRangeImage("rim1_6", {"image_shape": (180,180), "pan_range":(-jnp.pi, jnp.pi), "tilt_range":(-jnp.pi/2,jnp.pi/2)})

    field_shape = (60,180,20)
    origin = (-0.10, -1.99, -0.03)
    origin = (0.10, -0.99, 0.04)
    field_units_per_meter = (100., 100., 100.)
    fm1 = VectorsToField("fm1", params={"field_shape": field_shape, "origin": origin, "field_units_per_meter": field_units_per_meter})
    fm2 = FieldToVectors("fm2", params={"origin": origin, "field_units_per_meter": field_units_per_meter})
    comp = CompressAxes("comp", params={"axis":(2,), "compression_type": "Maximum"})


    tcp_reader = TCPReader("tcp_reader", {"mode": "read", "ip": "127.0.0.1", "port": 50025, "shape": (299,448), 'time_step': 0.02})
    tcp_reader_joints = TCPReader("tcp_reader_joints", {"mode": "read", "ip": "127.0.0.1", "port": 50022, "shape": (2,1), 'time_step': 0.02})

    tcp_lidar_writer = TCPWriter("tcp_lidar_writer", {"ip": "127.0.0.1", "port": 50001, "shape": (299,448), 'time_step': 0.02})
    tcp_head_writer = TCPWriter("tcp_head_writer", {"ip": "127.0.0.1", "port": 50002, "shape": (180,180), 'time_step': 0.02})
    tcp_allo_writer1 = TCPWriter("tcp_writer1", {"ip": "127.0.0.1", "port": 500123, "shape": (180,180), 'time_step': 0.02})
    tcp_allo_writer = TCPWriter("tcp_writer", {"ip": "127.0.0.1", "port": 50003, "shape": (60,180,20), 'time_step': 0.02})
    tcp_allo_writer2 = TCPWriter("tcp_writer2", {"ip": "127.0.0.1", "port": 50004, "shape": (60,180,20), 'time_step': 0.02})


    nf1 = NeuralField("nf1", {"shape": (180,180), "sigmoid": "AbsSigmoid", "beta": 100, "theta": 0, "resting_level": -5, "global_inhibition": 0, "input_noise_gain": 0, "tau": 0.1})
    nf2 = NeuralField("nf2", {"shape": (299,448), "sigmoid": "AbsSigmoid", "beta": 100, "theta": 0, "resting_level": -5, "global_inhibition": 0, "input_noise_gain": 0, "tau": 0.1})
    nf3 = NeuralField("nf3", {"shape": (180,180), "sigmoid": "AbsSigmoid", "beta": 100, "theta": 0, "resting_level": -5, "global_inhibition": 0, "input_noise_gain": 0, "tau": 0.1})
    nf4 = NeuralField("nf4", {"shape": (60,180), "sigmoid": "AbsSigmoid", "beta": 100, "theta": 0, "resting_level": -5, "global_inhibition": 0, "input_noise_gain": 0, "tau": 0.1})
    nf5 = NeuralField("nf5", {"shape": (60,180,20), "sigmoid": "AbsSigmoid", "beta": 100, "theta": 0, "resting_level": -5, "global_inhibition": 0, "input_noise_gain": 0, "tau": 0.1})



    tcp_reader >> tcp_lidar_writer >> nf2
    tcp_reader >> img_proj >> rim1 >> rim1_5 >> rim1_6 >> tcp_head_writer >> nf3

    img_proj >> coord_transf1 >> rim3 >> tcp_allo_writer1 >> nf1
    tcp_reader_joints >> "coord_transf1.in1" 

    coord_transf1 >> coord_transf2 >> rim2 >> nf1
    tcp_reader_joints >> "coord_transf2.in1" 

    img_proj >> coord_transf3 >> rim4 >> nf1
    tcp_reader_joints >> "coord_transf3.in1" 

    coord_transf1 >> fm1 >> comp >> nf4
    fm1 >> tcp_allo_writer

    fm1 >> fm2 >> coord_transf4 >> rim5 >> nf3
    tcp_reader_joints >> "coord_transf4.in1" 
    fm2 >> tcp_allo_writer2

    


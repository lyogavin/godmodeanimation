
from __future__ import print_function
import traceback
import bpy
import time
from tqdm import tqdm
import math
import bpy
import time
from tqdm import tqdm
import random
import time
import contextlib
import sys
import logging

import json
from pathlib import Path
from tqdm import tqdm
import os
from PIL import Image
import io
import base64
from mathutils import *

C = bpy.context
D = bpy.data



# Create a custom logger
logger = logging.getLogger(__name__)



def setup_cycles_rendering():
    bpy.data.scenes[0].render.engine = "CYCLES"

    # Set the device_type
    bpy.context.preferences.addons[
      "cycles"
    ].preferences.compute_device_type = "CUDA" # or "OPENCL"

    # Set the device and feature set
    bpy.context.scene.cycles.device = "GPU"

    # get_devices() to let Blender detects GPU device
    bpy.context.preferences.addons["cycles"].preferences.get_devices()
    logger.info(bpy.context.preferences.addons["cycles"].preferences.compute_device_type)
    for d in bpy.context.preferences.addons["cycles"].preferences.devices:
        d["use"] = 1 # Using all devices, include GPU and CPU
        logger.info((d["name"], d["use"]))

#setup_cycles_rendering()

def delete_all_cameras():
    for obj in bpy.data.objects:
        if 'camera' in obj.name.lower():
            if obj.name != "Camera":
                bpy.data.objects.remove(obj)

def delete_all_lights():
    for obj in bpy.data.objects:
        if 'light' in obj.name.lower() or 'area' in obj.name.lower():
            if obj.name != "Area":
                bpy.data.objects.remove(obj)

def delete_all_ojbs():
  if len(bpy.data.objects) == 0:
    return
  try:
    bpy.ops.object.mode_set(mode='OBJECT')
  except:
      pass
  try:
      bpy.ops.object.select_all(action='DESELECT')
  except:
      pass

  for obj in bpy.data.objects:
    #logger.info(obj)
    #if obj.name in bpy.data.objects:
    bpy.data.objects[obj.name].select_set(True)
    try:
      bpy.ops.object.delete()
    except Exception as e:
      #pass
      raise e


    return delete_all_ojbs()

#delete_all_ojbs()

def add_animation(animation_fbx):
    # make sure current object's bones are correctly named first.

    bpy.ops.import_scene.fbx(filepath=animation_fbx)

    delete_all_lights()
    delete_all_cameras()

    # set action
    D.objects['Armature'].animation_data.action = bpy.data.actions.get('Armature.001|mixamo.com|Layer0')

    # delete objects
    for obj in D.objects['Armature.001'].children:
        bpy.data.objects.remove(obj)

    for obj in bpy.data.objects:
        if 'Armature.001' == obj.name:
            bpy.data.objects.remove(obj)

    # go edit mode then go back to obj mode to trigger

    try:
        bpy.ops.object.select_all(action='DESELECT')
    except:
        pass


    bpy.data.objects["Armature"].select_set(True)

    bpy.context.view_layer.objects.active = bpy.data.objects['Armature']

    try:
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.object.mode_set(mode='OBJECT')

    except:
        pass


def rename_bones():
    for ob in D.objects['Armature'].pose.bones:
        if len(ob.name.split(":")) > 1 and ob.name.startswith("mixamorig"):
            ob.name = f"mixamorig:" + ob.name.split(":")[1]



def constraint_camera_to_armature(act=None):
    NEEDS_CONSTRAINTS = ['Jump1', "Jump2", "Running Jump", "Fast Run", "Running", "Standard Run", "Sprint", 'Standard Walk', 'Walking']

    if act is not None and  (not act in NEEDS_CONSTRAINTS):
        return
    bpy.context.view_layer.objects.active = D.objects['Camera']
    bpy.ops.object.constraint_add(type='CHILD_OF')
    D.objects['Camera'].constraints['Child Of'].target = D.objects['Armature']
    D.objects['Camera'].constraints['Child Of'].subtarget = 'mixamorig:Head'
    D.objects['Camera'].constraints['Child Of'].use_location_x = False
    D.objects['Camera'].constraints['Child Of'].use_location_y = True
    D.objects['Camera'].constraints['Child Of'].use_location_z = False
    D.objects['Camera'].constraints['Child Of'].use_rotation_x = False
    D.objects['Camera'].constraints['Child Of'].use_rotation_y = False
    D.objects['Camera'].constraints['Child Of'].use_rotation_z = False
    D.objects['Camera'].constraints['Child Of'].use_scale_x = False
    D.objects['Camera'].constraints['Child Of'].use_scale_y = False
    D.objects['Camera'].constraints['Child Of'].use_scale_z = False

def load_sword(sword_path, obj_name='Maria_sword'):
    # sword_fxb_path
    # with contextlib.redirect_stdout(None), contextlib.redirect_stderr(None):
    bpy.ops.import_scene.fbx(filepath=sword_path)
    bpy.ops.object.mode_set(mode='OBJECT')
    # set child of constraint
    bpy.context.view_layer.objects.active = D.objects[obj_name]
    bpy.ops.object.constraint_add(type='CHILD_OF')
    bpy.data.objects[obj_name].constraints['Child Of'].target = bpy.data.objects['Armature']
    bpy.data.objects[obj_name].constraints['Child Of'].subtarget = 'mixamorig:RightHand'
    # set location/rotation
    # D.objects[obj_name].location = Vector((0.0, 1.3969838619232178e-09, 4.656612873077393e-10))
    # D.objects[obj_name].rotation_euler = Euler((-10.340351104736328, 3.973609209060669, 0.9622434377670288), 'XYZ')
    D.objects[obj_name].location = Vector((0.0, 1.3969838619232178e-09, 4.656612873077393e-10))
    D.objects[obj_name].rotation_euler = Euler((-9.908905029296875, 3.6032626628875732, 0.8878166079521179),
                                                    'XYZ')
    # set inverse matrix:
    cons = bpy.data.objects[obj_name].constraints['Child Of']
    cons.inverse_matrix = Matrix(((100.0, -0.0, 0.0, -0.0),
                                  (-0.0, 7.549789643235272e-06, 100.0, 0.0),
                                  (0.0, -100.0, 7.549789643235272e-06, -0.0),
                                  (0.0, 0.0, -0.0, 1.0)))
    # set sword color
    matg = bpy.data.materials.new("sword")
    matg.use_nodes = True
    tree = matg.node_tree
    nodes = tree.nodes
    bsdf = nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (0.2, 0.2, 0.2, 0.8)
    matg.diffuse_color = (0.2, 0.2, 0.2, 0.8)
    D.objects[obj_name].active_material = matg
    # move to hand tal
    if "mixamorig:RightHand" not in D.objects['Armature'].data.bones:
        logger.warn(f"no right hand found in armature! skippping")
        return False
    else:
        D.objects[obj_name].location[2] = D.objects[obj_name].location[2] - (
                D.objects['Armature'].data.bones['mixamorig:RightHand'].tail_local -
                D.objects['Armature'].data.bones[
                    'mixamorig:RightHand'].head_local)[0] / 100

        return True


def add_camera(act):
    act_to_camera_params = {
        'Mma Kick': {
            'position': (-15.3823, -0.78685, 2.438),
            'rotation': (83.9261 * math.pi / 180.0, 0.828995 * math.pi / 180.0, -89.1498 * math.pi / 180.0),
            'lens': 82,
            'world_background': 5.2
        },
        'Kicking': {
            'position': (-10.371965408325195, -1.1039546728134155, 1.4159834384918213),
            'rotation': (1.5289082527160645, 9.589198270987254e-07, -1.5707974433898926),
            'lens': 50,
            'world_background': 5.2
        },
        'Great Sword Slash': {
            'position': (-10.371965408325195, -2.4149668216705322, 1.4159834384918213),
            'rotation': (1.5289082527160645, 9.589198270987254e-07, -1.5707974433898926),
            'lens': 50,
            'world_background': 5.2
        },
        'Great Sword Slash1': {
            'position': (-10.371965408325195, -1.4155880212783813, 1.4159834384918213),
            'rotation': (1.5289082527160645, 9.589198270987254e-07, -1.5707974433898926),
            'lens': 50,
            'world_background': 5.2
        },
        'Great Sword Jump Attack': {
            'position': (-10.371965408325195, -2.4149668216705322, 1.4159834384918213),
            'rotation': (1.5289082527160645, 9.589198270987254e-07, -1.5707974433898926),
            'lens': 50,
            'world_background': 5.2
        },
        'Chapa Giratoria 2': {
            'rotation': (1.4451348781585693, -7.169876994339575e-07, -3.1765003204345703),
            'position': (1.149894118309021, 8.868803024291992, 2.2082724571228027),
            'lens': 50,
            'world_background': 5.2
        },
        'Mma Kick Still Picture': {
            'position': (-0.19298392534255981, -7.772148132324219, 1.8676466941833496),
            'rotation': (1.4368618726730347, 0.014590280130505562, -0.041164956986904144),
            'lens': 50,
            'world_background': 5.2
        },
        'Great Sword High Spin Attack': {
            'position': (-10.371965408325195, -1.4155880212783813, 1.4159834384918213),
            'rotation': (1.5289082527160645, 9.589198270987254e-07, -1.5707974433898926),
            'lens': 50,
            'world_background': 5.2
        },
        'Fast Run': {
            'position': (-10.371965408325195, 0.006193757057189941, 1.4159834384918213),
            'rotation': (1.5289082527160645, 9.589198270987254e-07, -1.5707974433898926),
            'lens': 50,
            'world_background': 5.2
        },
        'Running': {
            'position': (-10.371965408325195, 0.006193757057189941, 1.4159834384918213),
            'rotation': (1.5289082527160645, 9.589198270987254e-07, -1.5707974433898926),
            'lens': 50,
            'world_background': 5.2
        },
        'Standard Run': {
            'position': (-10.371965408325195, 0.006193757057189941, 1.4159834384918213),
            'rotation': (1.5289082527160645, 9.589198270987254e-07, -1.5707974433898926),
            'lens': 50,
            'world_background': 5.2
        },
        'Sprint': {
            'position': (-10.371965408325195, 0.006193757057189941, 1.4159834384918213),
            'rotation': (1.5289082527160645, 9.589198270987254e-07, -1.5707974433898926),
            'lens': 50,
            'world_background': 5.2
        },
        'Idle': {
            'position': (-10.371965408325195, 0.006193757057189941, 1.4159834384918213),
            'rotation': (1.5289082527160645, 9.589198270987254e-07, -1.5707974433898926),
            'lens': 50,
            'world_background': 5.2
        },
        'Jump': {
            'position': (-10.371965408325195, 0.006193757057189941, 1.4159834384918213),
            'rotation': (1.5289082527160645, 9.589198270987254e-07, -1.5707974433898926),
            'lens': 50,
            'world_background': 5.2
        },
        'Running Jump': {
            'position': (-10.371965408325195, 0.006193757057189941, 1.4159834384918213),
            'rotation': (1.5289082527160645, 9.589198270987254e-07, -1.5707974433898926),
            'lens': 50,
            'world_background': 5.2
        },
        'Jump1': {
            'position': (-10.371965408325195, 0.006193757057189941, 1.4159834384918213),
            'rotation': (1.5289082527160645, 9.589198270987254e-07, -1.5707974433898926),
            'lens': 50,
            'world_background': 5.2
        },
        'Jump2': {
            'position': (-10.371965408325195, 0.006193757057189941, 1.4159834384918213),
            'rotation': (1.5289082527160645, 9.589198270987254e-07, -1.5707974433898926),
            'lens': 50,
            'world_background': 5.2
        },
        'Jumping': {
            'position': (-10.371965408325195, 0.006193757057189941, 1.4159834384918213),
            'rotation': (1.5289082527160645, 9.589198270987254e-07, -1.5707974433898926),
            'lens': 50,
            'world_background': 5.2
        },
    }
    camera_position = (0.79555, -7.4316, 0.80254)
    camera_rotation = (90. * math.pi / 180.0, 0.0 * math.pi / 180.0, 0.0 * math.pi / 180.0)
    camera_lens = 50.0
    world_bg = 5.2
    if act not in act_to_camera_params:
        act = 'Running'
    if act in act_to_camera_params:
        logger.info(f"found params for {act}, applying: {act_to_camera_params[act]}")
        camera_position = act_to_camera_params[act]['position']
        camera_rotation = act_to_camera_params[act]['rotation']
        camera_lens = act_to_camera_params[act]['lens']
        world_bg = act_to_camera_params[act]['world_background']
    bpy.ops.object.camera_add(enter_editmode=False, align='VIEW',
                              location=camera_position,
                              # rotation=camera_rotation
                              )
    camera = bpy.context.object
    camera.rotation_mode = 'XYZ'
    camera.rotation_euler = camera_rotation
    camera.location.x = camera_position[0]
    camera.location.y = camera_position[1]
    camera.location.z = camera_position[2]
    camera.data.lens = camera_lens  # .0
    camera.data.lens_unit = 'MILLIMETERS'
    bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[1].default_value = world_bg


    # set world bg color:
    # hard to remove the edge trace
    #bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[0].default_value = (0.051, 0.00481, 0.0344, 1.0)


    bpy.context.scene.camera = camera
    assert 'Camera' in [x.name for x in bpy.data.objects]
    return act_to_camera_params[act] if act in act_to_camera_params else None



# add light


def add_light(act):
    act_to_light_params = {
        'Mma Kick': {
            'location': (1.07015, -0.155719, 2.46545),
            'rotation': (37.261 * math.pi / 180.0,
                         -7.19536 * math.pi / 180.0,
                         99.1081 * math.pi / 180.0),
            'world_strength': 5.2,
            'energy': 2000,
            'size': 3.25,
            'size_y': 3.25
        },
        'Kicking': {
            'location': (1.3349692821502686, -0.5992897748947144, 3.5859198570251465),
            'rotation': (0.6503279805183411, 0.055217113345861435, 1.8663908243179321),
            'world_strength': 5.2,
            'energy': 2000,
            'size': 3.25,
            'size_y': 3.25
        },
        'Great Sword Slash': {
            'location': (2.2827718257904053, -2.786142349243164, 3.8406260013580322),
            'rotation': (0.6225225329399109, -0.00405377522110939, 1.5557596683502197),
            'world_strength': 5.2,
            'energy': 2000,
            'size': 3.25,
            'size_y': 3.25
        },
        'Great Sword Slash1': {
            'location': (2.2827718257904053, -2.786142349243164, 3.8406260013580322),
            'rotation': (0.6225225329399109, -0.00405377522110939, 1.5557596683502197),
            'world_strength': 5.2,
            'energy': 2000,
            'size': 3.25,
            'size_y': 3.25
        },
        'Great Sword Jump Attack': {
            'location': (2.2827718257904053, -2.786142349243164, 3.8406260013580322),
            'rotation': (0.6225225329399109, -0.00405377522110939, 1.5557596683502197),
            'world_strength': 5.2,
            'energy': 2000,
            'size': 3.25,
            'size_y': 3.25
        },
        'Chapa Giratoria 2': {
            'location': (0.36691832542419434, -1.99949312210083, 2.4365248680114746),
            'rotation': (0.597967267036438, -6.315235137939453, 0.06870156526565552),
            'world_strength': 5.2,
            'energy': 2000,
            'size': 3.25,
            'size_y': 3.25
        },
        'Mma Kick Still Picture': {
            'location': (-0.5337939262390137, 0.5151256322860718, 2.121335506439209),
            'rotation': (0.27648842334747314, 0.3575495183467865, 2.1721463203430176),
            'world_strength': 5.2,
            'energy': 2000,
            'size': 3.25,
            'size_y': 3.25
        },
        'Great Sword High Spin Attack': {
            'location': (2.2827718257904053, -2.786142349243164, 3.8406260013580322),
            'rotation': (0.6225225329399109, -0.00405377522110939, 1.5557596683502197),
            'world_strength': 5.2,
            'energy': 2000,
            'size': 3.25,
            'size_y': 3.25
        },
        'Fast Run': {
            'location': (2.2827718257904053, -0.05426359176635742, 3.8406260013580322),
            'rotation': (0.6225225329399109, -0.00405377522110939, 1.5557596683502197),
            'world_strength': 5.2,
            'energy': 2000,
            'size': 14,
            'size_y': 3.25
        },
        'Running': {
            'location': (-5.5996599197387695, 0.487104594707489, 0.40277886390686035),
            'rotation': (-1.498121976852417, -0.08005490899085999, 1.5331082344055176),
            'world_strength': 5.2,
            'energy': 2000,
            'size': 8.35,
            'size_y': 8.25
        },
        'Standard Run': {
            'location': (2.2827718257904053, -0.05426359176635742, 3.8406260013580322),
            'rotation': (0.6225225329399109, -0.00405377522110939, 1.5557596683502197),
            'world_strength': 5.2,
            'energy': 2000,
            'size': 14,
            'size_y': 3.25
        },
        'Sprint': {
            'location': (2.2827718257904053, -0.05426359176635742, 3.8406260013580322),
            'rotation': (0.6225225329399109, -0.00405377522110939, 1.5557596683502197),
            'world_strength': 5.2,
            'energy': 2000,
            'size': 14,
            'size_y': 3.25
        },
        'Idle': {
            'location': (2.2827718257904053, -0.05426359176635742, 3.8406260013580322),
            'rotation': (0.6225225329399109, -0.00405377522110939, 1.5557596683502197),
            'world_strength': 5.2,
            'energy': 2000,
            'size': 3.25,
            'size_y': 3.25
        },
        'Jumping': {
            'location': (2.2827718257904053, -0.05426359176635742, 3.8406260013580322),
            'rotation': (0.6225225329399109, -0.00405377522110939, 1.5557596683502197),
            'world_strength': 5.2,
            'energy': 2000,
            'size': 14,
            'size_y': 3.25
        },
        'Jump': {
            'location': (2.2827718257904053, -0.05426359176635742, 3.8406260013580322),
            'rotation': (0.6225225329399109, -0.00405377522110939, 1.5557596683502197),
            'world_strength': 5.2,
            'energy': 2000,
            'size': 14,
            'size_y': 3.25
        },
        'Running Jump': {
            'location': (2.2827718257904053, -0.05426359176635742, 3.8406260013580322),
            'rotation': (0.6225225329399109, -0.00405377522110939, 1.5557596683502197),
            'world_strength': 5.2,
            'energy': 2000,
            'size': 14,
            'size_y': 3.25
        },
        'Jump1': {
            'location': (2.2827718257904053, -0.05426359176635742, 3.8406260013580322),
            'rotation': (0.6225225329399109, -0.00405377522110939, 1.5557596683502197),
            'world_strength': 5.2,
            'energy': 2000,
            'size': 14,
            'size_y': 3.25
        },
        'Jump2': {
            'location': (2.2827718257904053, -0.05426359176635742, 3.8406260013580322),
            'rotation': (0.6225225329399109, -0.00405377522110939, 1.5557596683502197),
            'world_strength': 5.2,
            'energy': 2000,
            'size': 14,
            'size_y': 3.25
        },
    }
    location = (0.16575074195861816, -1.4841883182525635, 3.7574384212493896)
    rotation = (-0.03236645087599754, -0.5997453927993774, 1.5804479122161865)
    world_strength = 5.2
    if act not in act_to_light_params:
        act = 'Running'
    logger.info(f"found params for {act}, applying: {act_to_light_params[act]}")
    location = act_to_light_params[act]['location']
    rotation = act_to_light_params[act]['rotation']
    world_strength = act_to_light_params[act]['world_strength']
    energy = act_to_light_params[act]['energy']
    size = act_to_light_params[act]['size']
    size_y = act_to_light_params[act]['size_y']
    bpy.ops.object.light_add(type='AREA')
    light = bpy.data.objects['Area']
    light.data.energy = energy
    light.location = location
    light.data.size = size
    light.data.size_y = size_y
    light.data.use_shadow=False
    light.rotation_mode = 'XYZ'
    light.rotation_euler = rotation
    # set world light strength
    bpy.context.scene.world.node_tree.nodes['Background'].inputs[1].default_value = world_strength
    assert 'Area' in [x.name for x in bpy.data.objects]
    return act_to_light_params[act]


act_to_back_act_params = {
    'Mma Kick': {
        'start': 1,
        'end': 123,
        'step': 8
    },
    'Kicking': {
        'start': 40,
        'end': 140,
        'step': 6
    },
    'Chapa Giratoria 2': {
        'start': 40,
        'end': 140,
        'step': 6
    },
    'Great Sword Slash': {
        'start': 4,
        'end': 100,
        'step': 6
    },
    'Great Sword Slash1': {
        'start': 1,
        'end': 80,
        'step': 5
    },
    'Great Sword Jump Attack': {
        'start': 18,
        'end': 82,
        'step': 4
    },
    'Great Sword High Spin Attack': {
        'start': 27,
        'end': 123,
        'step': 6
    },
    'Fast Run': {
        'start': 1,
        'end': 32,
        'step': 2
    },
    'Running': {
        'start': 1,
        'end': 32,
        'step': 2
    },
    'Standard Run': {
        'start': 1,
        'end': 48,
        'step': 3
    },
    'Sprint': {
        'start': 1,
        'end': 32,
        'step': 2
    },
    'Running Jump': {
        'start': 1,
        'end': 62,
        'step': 4
    },
    'Jump': {
        'start': 1,
        'end': 131,
        'step': 8
    },
    'Jump2': {
        'start': 1,
        'end': 64,
        'step': 4
    },
    'Jump1': {
        'start': 1,
        'end': 64,
        'step': 4
    },
    'Jumping': {
        'start': 1,
        'end': 112,
        'step': 7
    },


    'Walking': {
        'start': 1,
        'end': 58,
        'step': -1
    },
    'Two Handed Sword Death': {
        'start': 1,
        'end': 145,
        'step': -1
    },
    'Sword And Shield Slash': {
        'start': 1,
        'end': 147,
        'step': -1
    },
    'Standing To Crouch': {
        'start': 1,
        'end': 56,
        'step': -1
    },
    'Standard Walk': {
        'start': 1,
        'end': 71,
        'step': -1
    },
    'Roundhouse Kick': {
        'start': 1,
        'end': 151,
        'step': -1
    },
    'Idle': {
        'start': 1,
        'end': 118,
        'step': -1
    },
    'Hook Punch': {
        'start': 1,
        'end': 131,
        'step': -1
    },
    'Gunplay': {
        'start': 1,
        'end': 13,
        'step': -1
    },
    'Great Sword Slash2': {
        'start': 1,
        'end': 77,
        'step': -1
    },
    'Falling To Landing': {
        'start': 1,
        'end': 65,
        'step': -1
    },
    'Falling Back Death': {
        'start': 1,
        'end': 132,
        'step': -1
    },
    'Crouch To Standing': {
        'start': 1,
        'end': 56,
        'step': -1
    },
    'Cross Punch': {
        'start': 1,
        'end': 121,
        'step': -1
    },

}


def back_action(act):
    try:
        bpy.ops.object.select_all(action='DESELECT')
    except:
        pass

    bpy.data.objects['Armature'].select_set(True)
    bpy.context.view_layer.objects.active = bpy.data.objects['Armature']

    bpy.ops.object.mode_set(mode='POSE')
    assert act in act_to_back_act_params

    start = act_to_back_act_params[act]['start']
    end = act_to_back_act_params[act]['end']
    step = act_to_back_act_params[act]['step']

    if step == -1:
        # calculate step auto
        if end-start <= 16:
            step = 1
            end = start + 16 - 1
        else:
            step = math.ceil((end - start)/16)

    ret = bpy.ops.nla.bake(frame_start=start, frame_end=end, step=step, bake_types={'POSE'})

    assert list(ret)[0] == 'FINISHED'

    # assert bpy.context.scene.frame_start == 0

    # total_frames = 123 #bpy.context.scene.frame_end - bpy.context.scene.frame_start  + 1
    # sample_frames = 16

    skip_frames = 0  # random.randint(0, (total_frames // sample_frames // 2)) + bpy.context.scene.frame_start
    sample_idx = list(range(start, end + 1, step))

    return sample_idx, act_to_back_act_params[act]

def setup_rendering_engine():
  # Set the render settings
  bpy.context.scene.render.engine = 'BLENDER_EEVEE'
  #bpy.context.scene.render.engine = 'BLENDER_WORKBENCH'
  #bpy.context.scene.render.engine = 'CYCLES' #'BLENDER_WORKBENCH' #BLENDER_EEVEE'


def render_and_save(frame_id, save_path):


  bpy.context.scene.frame_current = frame_id

  bpy.context.scene.render.filepath = save_path
  bpy.context.scene.render.image_settings.file_format = 'PNG'
  bpy.ops.render.render(animation=False, write_still=True)

def add_simple_toon_shader():
    setup_rendering_engine()
    for child in D.objects['Armature'].children:
        nodes = child.data.materials[0].node_tree.nodes
        links = child.data.materials[0].node_tree.links
        # find all the children with texture setting
        # find link node to remove
        to_remove_i = None
        for i in range(len(links.items())):
            if not "Image Texture" in nodes:
                continue
            if not "Principled BSDF" in nodes:
                continue
            if links[i].to_node == nodes["Principled BSDF"] and links[i].from_node == nodes["Image Texture"]:
                to_remove_i = i
        if to_remove_i is None:
            # no texture found, skipping
            continue
        #if to_remove_i is not None:
        links.remove(links[to_remove_i])
        # existing shader nodes:
        pbsdf = nodes["Principled BSDF"]
        # new shader nodes:
        strgb = nodes.new(type="ShaderNodeShaderToRGB")
        colorramp = nodes.new(type="ShaderNodeValToRGB")
        colorramp.color_ramp.interpolation = 'CONSTANT'
        colorramp.color_ramp.elements[1].position = 0.286
        colorramp.color_ramp.elements[0].color[0] = 0.5
        colorramp.color_ramp.elements[0].color[1] = 0.5
        colorramp.color_ramp.elements[0].color[2] = 0.5
        colorramp.color_ramp.elements[0].color[3] = 1.0
        mixrgb = nodes.new(type="ShaderNodeMixRGB")
        mixrgb.blend_type='MULTIPLY'
        # links:
        links.new(pbsdf.outputs['BSDF'], strgb.inputs['Shader'])
        links.new(strgb.outputs['Color'],colorramp.inputs['Fac'])
        links.new(colorramp.outputs['Color'], mixrgb.inputs['Color2'])
        links.new(mixrgb.outputs['Color'], nodes["Material Output"].inputs['Surface'])
        links.new(nodes["Image Texture"].outputs['Color'], mixrgb.inputs['Color1'])


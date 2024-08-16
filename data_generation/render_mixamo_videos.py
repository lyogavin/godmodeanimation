
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
from glob import glob


ts = int(time.time())


# Create a custom logger
logger = logging.getLogger(__name__)

f_handler = logging.FileHandler('render_mixamo_videos.log')
f_handler.setLevel(logging.INFO)
f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
f_handler.setFormatter(f_format)
logger.addHandler(f_handler)
logger.setLevel(logging.INFO)


#exit(1)

import bpy
from mathutils import *

C = bpy.context
D = bpy.data

from bpy_utils import (setup_cycles_rendering,
                        rename_bones,
                       delete_all_cameras,
                       delete_all_lights,
                        delete_all_ojbs,
                        add_animation,
                        load_sword,
                        add_camera,
                        add_light,
                        back_action,
                        setup_rendering_engine,
                        render_and_save,
                        constraint_camera_to_armature,
                        add_simple_toon_shader
                       )





import multiprocessing
from multiprocessing import Pool

def f(x):
    return x*x




import hashlib

random.seed(123)
def get_videoid_hash(title, act_title):
    h = hashlib.new('sha256')  # sha256 can be replaced with diffrent algorithms
    to_hash = f"{title}__{act_title}"

    h.update(to_hash.encode())  # give a encoded string. Makes the String to the Hash

    hex = str(h.hexdigest())
    return hex[:16]








def render_images_for_fbx(fbx_fn, action_fbx, title, act_title,videoid, to_save_dir, save_blend, use_toon_shader):
    if use_toon_shader:
        logger.info(f"use_toon_shader: {use_toon_shader}")
    setup_cycles_rendering()
    setup_rendering_engine()


    logger.info(f"processing {fbx_fn}")

    os.makedirs(to_save_dir, exist_ok=True)

    delete_all_ojbs()


    # set up camera light
    light_params = add_light('Running')


    # load new fbx
    with contextlib.redirect_stdout(None), contextlib.redirect_stderr(None):
        bpy.ops.import_scene.fbx(filepath=fbx_fn)
    rename_bones()



    # load sword
    LOAD_SWORD = False

    loaded_weapon = None

    if LOAD_SWORD:
        sword_fxb_path = "/root/downloaded_gameobject_models/sword_recenter_fix_camera.fbx"
        axe_fxb_path = "/root/downloaded_gameobject_models/brute_axe_fix_camera.fbx"


        load_weapon_id = random.randint(0,1)

        if load_weapon_id == 0:
            loaded_weapon = sword_fxb_path
            res = load_sword(sword_fxb_path)
        else:
            loaded_weapon = axe_fxb_path
            res = load_sword(axe_fxb_path, 'BattleAxe')



        if not res:
            logger.warn(f"error loading sword for {fbx_fn}, skipping...")
            return


    # setup animation action
    add_animation(action_fbx)

    cam_parmas = add_camera('Running')#act_title)
    constraint_camera_to_armature(act_title)


    # bake action
    sample_idx, bake_act_params = back_action(act_title)

    # apply toon shader
    if use_toon_shader:
        add_simple_toon_shader()




    saved_frame_paths = []




    for ino, render_i in tqdm(enumerate(sample_idx), total=len(sample_idx)):
        saved_frame = os.path.join(to_save_dir, f"v_{videoid}_{ino:04d}.png")
        if (not os.path.exists(saved_frame)) or (not check_image(saved_frame)):
            with contextlib.redirect_stdout(None), contextlib.redirect_stderr(None):
                render_and_save(render_i, saved_frame)
            logger.info(f"generating {ino} - {saved_frame}")
            logger.info(f"dump objs for {act_title}:")
            for obj in D.objects:
                logger.info(f"obj: {obj}")
                logger.info(f"obj location: {obj.location}")
                logger.info(f"obj rotation_euler: {obj.rotation_euler}")
        else:
            logger.info(f"already generated {saved_frame}")
        saved_frame_paths.append(saved_frame)


    try:
        sample_gend_path = f"sample_{title.replace(' ', '_').replace('/', '_').lower()}__{act_title.replace(' ', '_').replace('/', '_').lower()}.gif"
        frs = []
        for fr_path in saved_frame_paths:
            frs.append(Image.open(fr_path))

        sample_gend_path = os.path.join(to_save_dir, sample_gend_path)
        frs[0].save(sample_gend_path, save_all=True, append_images=frs[1:], duration=150, loop=0, format="gif")
    except Exception as e:
        logger.info(f"error: {e}", exc_info=True)

    # save meta to json
    meta = {'bake_act_params': bake_act_params,
            'light_params': light_params,
            'title': title,
            'act_title': act_title,
            'shader':'simple_toon' if use_toon_shader else 'default',
            'action_fbx': action_fbx,
            'videoid': videoid,
            'constraint_camera': True,
            'saved_files': saved_frame_paths,
            'loaded_weapon':loaded_weapon,
            'gend_gif': sample_gend_path,
            'cam_parmas': cam_parmas}

    render_params_file = os.path.join(to_save_dir, 'render_params.json')

    with open(render_params_file, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False)

    if save_blend:
        blend_save_path = os.path.join(to_save_dir, "render_blend.blend")
        bpy.ops.wm.save_as_mainfile(filepath=blend_save_path)



def mute():
    sys.stdout = open(os.devnull, 'w')

class Process(multiprocessing.Process):
    def __init__(self, *args, **kwargs):
        multiprocessing.Process.__init__(self, *args, **kwargs)
        self._pconn, self._cconn = multiprocessing.Pipe()
        self._exception = None

    def run(self):
        try:
            multiprocessing.Process.run(self)
            self._cconn.send(None)
        except Exception as e:
            logger.info(f"bpy info:")
            for obj in bpy.data.objects:
                logger.info(f"bpy obj: {obj}")
            logger.info(f"active: {bpy.context.view_layer.objects.active}")
            tb = traceback.format_exc()
            self._cconn.send((e, tb))
            # raise e  # You can still rise this exception if you need to

    @property
    def exception(self):
        if self._pconn.poll():
            self._exception = self._pconn.recv()
        return self._exception

def check_image(image_path):
    try:
        im = Image.open(image_path)
        im.verify()
        return True
    except Exception:
        return False

if __name__ == '__main__':

    # labeled list should not use toon shader:
    no_toon_shader_list = "Big Vegas,Aj,Skeletonzombie T Avelange,Paladin J Nordstrom,Jones,Chad,Ely By K.Atienza,Crypto,Kaya,Claire,Vampire A Lusth,Nightshade J Friedrich,Jennifer,Sporty Granny,Maw J Laygo,Alien Soldier,Eve By J.Gonzales,Drake"
    import argparse
    import PIL
    parser = argparse.ArgumentParser()
    parser.add_argument("--render_type", type=str, default="actions")
    parser.add_argument("--render_id", type=int, default=0)
    parser.add_argument("--render_id_total", type=int, default=1)
    parser.add_argument("--pool_size", type=int, default=8)
    parser.add_argument("--save_blend",  action='store_true')
    parser.add_argument("--no_toon_shader",  action='store_true')
    parser.add_argument("--no_toon_blender_titles",  type=str, default=no_toon_shader_list)
    parser.add_argument("--run_act_titles",  type=str, default=None)
    parser.add_argument("--run_char_titles",  type=str, default=None)
    parser.add_argument("--fbx_files_base", help="base dir for character fbx files",  type=str, default='/root/downloaded_chars_tpose/downloaded_fbx_files/')
    parser.add_argument("--download_json_path", help="json file for the meta data of the character fbx files", type=str, 
        default='/root/downloaded_chars_tpose/mixamo_downloads_chars_1711827999.jsonl')
    parser.add_argument("--download_actions_path_full_acts", help="base dir for action fbx files", type=str, 
        default='/root/downloaded_action_models/full_acts/')
    parser.add_argument("--gend_images_base", help="base dir for generated videos", type=str, default='/root/full_acts/gend_images_{ts}/full_acts_fix_shader_fix_light/')
    parser.add_argument("--run_act_to_exclude", help="comma separated list of actions to exclude", type=str, default=None)
    #bpy.ops.wm.save_as_mainfile(filepath='pathToNewBlendFile.blend')
    args = parser.parse_args()

    fbx_files_base = args.fbx_files_base
    download_json_path = args.download_json_path 
    download_actions_path_full_acts = args.download_actions_path_full_acts
    gend_images_base = args.gend_images_base

    if args.run_act_to_exclude is not None:
        run_act_to_exclude = args.run_act_to_exclude.split(",")

    run_char_titles = args.run_char_titles



    no_toon_blender_titles = None

    if args.no_toon_blender_titles is not None:

        no_toon_blender_titles = args.no_toon_blender_titles.split(",")



    # render action videos
    assert args.render_type == "actions"

    already_done_videoid_list = set()
    already_done_info_list = []
    json_file_missing_list = []

    # 1. firstly find all already done dirs
    logger.info(f"1. find all already done video dirs")
    #f"v_{videoid}_{ino:04d}.png")
    with open(download_json_path, 'r') as jf:
        lines = [x for x in jf]


    if run_char_titles is not None:
        filtered_lines = []

        for l in lines:
            json_dat = json.loads(l)
            title = json_dat['title']
            if title in run_char_titles:
                filtered_lines.append(l)
                print(f"rendering title: {title}")
        lines = filtered_lines

    input(f"totally {len(lines)} lines")

    iline_to_random_act_ids = dict()

    act_fbx_list = [x for x in glob(f"{download_actions_path_full_acts}/*.fbx") if
                         Path(x).stem + ".fbx" not in run_act_to_exclude]

    if args.run_act_titles is not None:
        act_fbx_list = [x for x in act_fbx_list if Path(x).stem in args.run_act_titles]

    act_fbx_list.sort()

    input(f"totally {len(act_fbx_list)} acts")

    assigned_act_fbx_list = []

    for i,a in enumerate(act_fbx_list):
        if i % args.render_id_total == args.render_id:
            assigned_act_fbx_list.append(a)

    for i_line, line in tqdm(enumerate(lines)):
        json_dat = json.loads(line)
        # logger.info(json_dat)
        title = json_dat['title']


        for act_fbx in assigned_act_fbx_list:
            act_type = Path(act_fbx).stem
            act_title = Path(act_fbx).stem

            if act_title  + ".fbx" in run_act_to_exclude:
                logger.info(f"skipping {act_title} as in to_exclude list")
                continue
            #if act_title not in used_acts:
            #    continue
            fbx_fn = act_fbx
            #fbx_fn = os.path.join(fbx_files_base, fbx_fn + ".fbx")

            # info for logging:
            videoid = get_videoid_hash(title, act_title)
            to_save_dir = os.path.join(gend_images_base, f"v_{videoid}_dir")

            if os.path.exists(to_save_dir):
                logger.info(f"checking dir: {to_save_dir}")

            if not os.path.exists(os.path.join(to_save_dir, 'render_params.json')):
                json_file_missing_list.append(to_save_dir)

            if os.path.exists(os.path.join(to_save_dir, 'render_params.json')):

                render_params_file = os.path.join(to_save_dir, 'render_params.json')

                with open(render_params_file, 'r', encoding='utf-8') as f:
                    dat = json.load(f)

                if 'title' not in dat:
                    logger.info(f"adding title and act title in json for {videoid}")

                    dat['title'] = title
                    dat['act_title'] = act_title
                    dat['videoid'] = videoid
                    dat['shader'] = 'simple_toon'


                    dat['saved_files'] =  [os.path.join(to_save_dir, f"v_{videoid}_{i:04d}.png") for i in range(16)]

                    sample_gend_path = f"sample_{title.replace(' ', '_').replace('/', '_').lower()}__{act_title.replace(' ', '_').replace('/', '_').lower()}.gif"
                    sample_gend_path = os.path.join(to_save_dir, sample_gend_path)

                    dat['gend_gif'] = sample_gend_path


                    with open(render_params_file, 'w', encoding='utf-8') as f:
                        json.dump(dat, f, ensure_ascii=False)


            valid_count = 0
            for i in range(16):
                if os.path.exists(os.path.join(to_save_dir, f"v_{videoid}_{i:04d}.png")) and \
                        check_image(os.path.join(to_save_dir, f"v_{videoid}_{i:04d}.png")):
                    valid_count += 1
                elif os.path.exists(os.path.join(to_save_dir, f"v_{videoid}_{i:04d}.png")) and \
                        not check_image(os.path.join(to_save_dir, f"v_{videoid}_{i:04d}.png")):
                    logger.info(f"invalid image found: {os.path.join(to_save_dir, f'v_{videoid}_{i:04d}.png')}")
            if valid_count == 16 and os.path.exists(os.path.join(to_save_dir, 'render_params.json')):
                already_done_videoid_list.add(videoid)
                already_done_info_list.append(f"v_{videoid}_dir")
    logger.info(f"found already done video ids: {len(already_done_videoid_list)}")
    logger.info(f"found already done video dirs: {already_done_info_list}")
    logger.info(f"json missing dirs: {json_file_missing_list}")

    input('continue...')

    # 2. render videos
    to_render_list = []

    with open(download_json_path, 'r') as jf:
        lines = [x for x in jf]
        if run_char_titles is not None:
            filtered_lines = []

            for l in lines:
                json_dat = json.loads(l)
                title = json_dat['title']
                if title in run_char_titles:
                    filtered_lines.append(l)
                    print(f"rendering title: {title}")
            lines = filtered_lines

        random.shuffle(lines)
        for i_line, line in tqdm(enumerate(lines)):
            json_dat = json.loads(line)
            # logger.info(json_dat)
            title = json_dat['title']

            title_no_toon_blender = False

            if title in no_toon_blender_titles:
                title_no_toon_blender = True

            for act_fbx in assigned_act_fbx_list:
                act_type = Path(act_fbx).stem
                act_title = Path(act_fbx).stem
                if act_title  + ".fbx" in run_act_to_exclude:
                    print(f"skipping {act_title} as in to_exclude list")
                    continue
                # if act_title not in used_acts:
                #    continue
                #fbx_fn = act_fbx
                fbx_fn = json_dat['act_fbxs'][0]

                fbx_fn = os.path.join(fbx_files_base, Path(fbx_fn).stem + ".fbx")


                # info for logging:
                videoid = get_videoid_hash(title, act_title)
                to_save_dir = os.path.join(gend_images_base, f"v_{videoid}_dir")

                use_toon_shader = True
                if args.no_toon_shader:
                    use_toon_shader = False
                if title_no_toon_blender:
                    use_toon_shader = False

                if videoid not in already_done_videoid_list:
                    to_render_list.append((fbx_fn,act_fbx, title, act_title, videoid, to_save_dir, args.save_blend, use_toon_shader))

                # if act_title != 'Chapa Giratoria 2':
                #    continue
    POOL_SIZE = args.pool_size

    logger.info(f"to render: {len(to_render_list)}")


    for i in range(0, len(to_render_list), POOL_SIZE):
        logger.info(f"start rendering {i}th. of {len(to_render_list)} models.")
        ps = []

        start = time.time()
        for j in range(i, min(i+POOL_SIZE, len(to_render_list))):
            p = Process(target=render_images_for_fbx, args=to_render_list[j])

            p.start()
            ps.append((p, to_render_list[j]))

        for p in ps:

            p[0].join()
            if p[0].exception:
                error, traceback = p[0].exception
                logger.info(error)
                logger.info(traceback)
                sys.exit(1)
            logger.info(f"finished processing: {p[1]}")
        logger.info(f"rendered {POOL_SIZE} fbx acts in {time.time()-start:.02f} secs")



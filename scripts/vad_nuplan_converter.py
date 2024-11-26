import os
import math
import copy
import argparse
from os import path as osp
from collections import OrderedDict
from typing import List, Tuple, Union

from time import time
from matplotlib.pylab import f
from matplotlib.style import available, use
import numpy as np
from py import log
from pyquaternion import Quaternion
from shapely.geometry import MultiPoint, box
import uuid
from distutils.util import strtobool
from omegaconf import OmegaConf
from typing import Any, Generator, List, Optional, Set, Tuple, Type, cast, Dict

from nuplan.database.nuplan_db_orm.nuplandb import NuPlanDB
from nuplan.database.maps_db.gpkg_mapsdb import GPKGMapsDB
from nuplan.database.nuplan_db_orm.camera import Camera
from nuplan.database.nuplan_db_orm.ego_pose import EgoPose
from nuplan.database.nuplan_db_orm.image import Image
from nuplan.database.nuplan_db_orm.lidar import Lidar
from nuplan.database.nuplan_db_orm.lidar_box import LidarBox
from nuplan.database.nuplan_db_orm.lidar_pc import LidarPc
from nuplan.database.nuplan_db_orm.track import Track

from tqdm import tqdm
from collections import deque


################Configurations################
nuplan_categories = (
    "vehicle",
    "bicycle",
    "pedestrian",
    "traffic_cone",
    "barrier",
    "czone_sign",
    "generic_object",
)

ego_width, ego_length = 1.85, 4.084

FRAME_RATE_IMAGE = 10 # Frequency of the image in database
TARGET_FRAME_RATE = 2 # Target frequency in model training

##################Functions##################

def translate_lidar_box(lidar_box: LidarBox, translation: np.ndarray) -> LidarBox:
    lidar_box.x += translation[0]
    lidar_box.y += translation[1]
    lidar_box.z += translation[2]
    return lidar_box

def rotate_lidar_box(lidar_box: LidarBox, rotation: Quaternion) -> LidarBox:
    point = np.array([lidar_box.x, lidar_box.y, lidar_box.z])
    rotated_point = rotation.rotate(point)
    lidar_box.x, lidar_box.y, lidar_box.z = rotated_point

    current_orientation = Quaternion(axis=[0, 0, 1], angle=lidar_box.yaw)
    new_orientation = rotation * current_orientation
    lidar_box.yaw = new_orientation.yaw_pitch_roll[0]

    velocity = np.array([lidar_box.vx, lidar_box.vy, lidar_box.vz])
    rotated_velocity = rotation.rotate(velocity)
    lidar_box.vx, lidar_box.vy, lidar_box.vz = rotated_velocity

    return lidar_box

def quart_to_rpy(qua):
    x, y, z, w = qua
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(2 * (w * y - x * z))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (z * z + y * y))
    return roll, pitch, yaw

def yaw_to_quaternion(yaw):
    return Quaternion(axis=[0, 0, 1], angle=yaw)

def locate_message(utimes, utime):
    i = np.searchsorted(utimes, utime)
    if i == len(utimes) or (i > 0 and utime - utimes[i - 1] < utimes[i] - utime):
        i -= 1
    return i

def transform_matrix(translation, rotation, inverse=False):
    """
    Convert pose to transformation matrix.
    
    Args:
        translation (list or np.ndarray): Translation vector [x, y, z].
        rotation (Quaternion): Rotation in quaternion format.
        inverse (bool): If set to True, computes the inverse transform.
    
    Returns:
        np.ndarray: 4x4 transformation matrix.
    """
    translation = np.array(translation)
    rotation_matrix = rotation.rotation_matrix
    
    if inverse:
        rot_inv = rotation_matrix.T
        trans = rot_inv.dot(-translation)
        transform = np.eye(4)
        transform[:3, :3] = rot_inv
        transform[:3, 3] = trans
    else:
        transform = np.eye(4)
        transform[:3, :3] = rotation_matrix
        transform[:3, 3] = translation
    
    return transform

def create_nuplan_infos(
    root_path,
    out_path,
    version="trainval",
    use_multiprocessing=True
):
    """Create info file of nuPlan dataset.

    Given the raw data, generate its related info file in pkl format.

    Args:
        root_path (str): Path of the data root.
        out_path (str): Path to save the info file.
        split_file (str): Path of the split file.
        version (str): Version of the data.
            Default: 'trainval'
        max_sweeps (int): Max number of sweeps.
            Default: 10
    """

    # set the environment variable
    NUPLAN_DATA_ROOT = root_path
    NUPLAN_MAPS_ROOT = osp.join(NUPLAN_DATA_ROOT, "maps")
    NUPLAN_DB_FILES = osp.join(NUPLAN_DATA_ROOT, f"nuplan-v1.1/splits/{version}")
    NUPLAN_MAP_VERSION = "nuplan-maps-v1.0"
    NUPLAN_SENSOR_ROOT = osp.join(NUPLAN_DATA_ROOT, f"nuplan-v1.1/sensor_blobs")

    # set the environment variable
    os.environ["NUPLAN_DATA_ROOT"] = NUPLAN_DATA_ROOT
    os.environ["NUPLAN_MAPS_ROOT"] = NUPLAN_MAPS_ROOT
    os.environ["NUPLAN_DB_FILES"] = NUPLAN_DB_FILES
    os.environ["NUPLAN_MAP_VERSION"] = NUPLAN_MAP_VERSION
    os.environ["NUPLAN_SENSOR_ROOT"] = NUPLAN_SENSOR_ROOT

    # wait for the nuplan database to be ready
    print("waiting for the nuplan database to be ready")
    maps_db = GPKGMapsDB(
        map_version=NUPLAN_MAP_VERSION,
        map_root=NUPLAN_MAPS_ROOT,
    )
    print("nuplan database is ready")
    print(version, root_path)

    _, _ = _fill_trainval_infos(
        maps_db=maps_db, use_multiprocessing=use_multiprocessing, output_dir=out_path
    )



def _fill_trainval_infos(
    fut_ts=6,
    his_ts=2,
    maps_db=None,
    use_multiprocessing=True, 
    output_dir=None
):
    """Generate the train/val infos from the raw data.

    Args:
        split (dict): Split information loaded from the split file.
        version (str): Version of the data. Default: 'train'.
        max_sweeps (int): Max number of sweeps. Default: 10.

    Returns:
        tuple[list[dict]]: Information of training set and validation set
            that will be saved to the info file.
    """
    # init the variables
    nuplan_infos = []
    val_nuplan_infos = []
    cat2idx = {}
    for idx, name in enumerate(nuplan_categories):
        cat2idx[name] = idx

    splits = os.listdir(os.environ["NUPLAN_DB_FILES"])


    # TODO: downsample to 2hz，accelerate the speed of sql query
    def process_log(log_file_name, output_dir= None, location=None):   
        log_file_path = os.path.join(os.environ["NUPLAN_DB_FILES"], log_file_name)
        if osp.exists(log_file_path) is False or (not log_file_name.endswith(".db")):
            return

        if output_dir is None:
            print("Output dir is None, set to default: data/output")
            output_dir = os.path.join("data", "output")
            os.makedirs(output_dir, exist_ok=True)
        else:
            if os.path.exists(output_dir) is False:
                os.makedirs(output_dir, exist_ok=True)

        output_pkl = os.path.splitext(log_file_name)[0] + ".pkl"
        output_pkl_path = os.path.join(output_dir, output_pkl)

        if os.path.exists(output_pkl_path) and args.overwrite is False:
            return

        # load the log file
        nuplan_db = NuPlanDB(
            data_root=os.environ["NUPLAN_DATA_ROOT"],
            load_path=log_file_path,
            maps_db=maps_db,
        )
        map_location = nuplan_db.log.location
        if location is not None:
            if map_location != location:
                return
        
        print(f"Processing {log_file_name}...")

        # TODO: improve speed
        sample_token_buffer = dict()
        for i, image in enumerate(nuplan_db.image):         
            # image.lidar_pc is sd_rec
            # image -> lidar_pc -> other information
            lidar_pc_token = image.lidar_pc.token

            # find 8 camera images corresponding to the lidar_pc
            # the corresponded Lidar_pc&Images are sample data token
            # Lidar_pc contains the information of the ego pose and the boxes token


            if lidar_pc_token not in sample_token_buffer:
                sample_token_buffer[lidar_pc_token] = [i]
            else:
                sample_token_buffer[lidar_pc_token] += [i]

        # delete unmatched images
        deleted_token_list = []
        for lidar_pc_token in tqdm(sample_token_buffer, leave=False):
            if len(sample_token_buffer[lidar_pc_token]) != 8: # 8 cameras in total
                deleted_token_list.append(lidar_pc_token)
        
        for deleted_token in deleted_token_list:
            del sample_token_buffer[deleted_token]

        lidar_pc_token_list = list(sample_token_buffer.keys())
             
        lidar_pc_token_list = lidar_pc_token_list[::(FRAME_RATE_IMAGE // TARGET_FRAME_RATE)] # Downsample

        if len(lidar_pc_token_list) < his_ts + fut_ts + 2:
            print(f"Not enough timestamp to compute his/fut trajectory! Skip.\nFile name: {log_file_name}")
            # save pesudo data
            import pickle
            pickle.dump([], open(output_pkl_path, "wb"))
            return

        lidar_pc_deq = deque(maxlen=his_ts+1)
        fut_lidar_pc_deq = deque(maxlen=fut_ts+1)
        
        for i in range(his_ts+1):
            current_lidar_pc = nuplan_db.image[sample_token_buffer[lidar_pc_token_list[i]][0]].lidar_pc
            lidar_pc_deq.append(current_lidar_pc)
        
        for i in range(his_ts+1, his_ts+fut_ts+2):
            fut_lidar_pc = nuplan_db.image[sample_token_buffer[lidar_pc_token_list[i]][0]].lidar_pc
            fut_lidar_pc_deq.append(fut_lidar_pc)

        frame_idx = -1
        # keep the scene token the same for the same log file, because of its consistency
        scene_token = uuid.uuid5(uuid.NAMESPACE_DNS, log_file_name).hex

        
        for lidar_pc_token, fut_lidar_pc_token in tqdm(zip(lidar_pc_token_list[his_ts+1:-fut_ts], lidar_pc_token_list[his_ts+fut_ts+1:]), total=len(lidar_pc_token_list[his_ts+1:-fut_ts]), leave=False):
            frame_idx += 1

            cur_lidar_pc = nuplan_db.image[sample_token_buffer[lidar_pc_token][0]].lidar_pc

            lidar_pc_deq.append(cur_lidar_pc)

            fut_lidar_pc = nuplan_db.image[sample_token_buffer[fut_lidar_pc_token][0]].lidar_pc

            fut_lidar_pc_deq.append(fut_lidar_pc)

            image_f0 = Image()
            for image_id in sample_token_buffer[lidar_pc_token]:
                image_f0 = nuplan_db.image[image_id]
                if image_f0.camera.channel == "CAM_F0":
                    break

            cs_record = nuplan_db.lidar.get(image_f0.lidar_pc.lidar_token)
            pose_record = nuplan_db.ego_pose.get(image_f0.lidar_pc.ego_pose_token)

            # get the lidar path
            lidar_path = os.path.join(
                os.environ["NUPLAN_SENSOR_ROOT"], image_f0.lidar_pc.filename
            )

            roadblock_ids = cur_lidar_pc.scene.roadblock_ids
            roadblock_ids = [int(id) for id in roadblock_ids.split()]

            # can_bus is in the ego_pose
            # refer to: https://lightwheel.feishu.cn/wiki/Hq5WwooPfiy4Aik9kSjclF5FnHc
            pose = copy.deepcopy(pose_record)
            yaw_rad = math.atan2(2 * (pose.qw * pose.qz + pose.qx * pose.qy), 1 - 2 * (pose.qy**2 + pose.qz**2))
            yaw_deg = math.degrees(yaw_rad)

            can_bus = np.array([
                pose.x, pose.y, pose.z, 
                pose.qx, pose.qy, pose.qz, pose.qw, 
                pose.acceleration_x, pose.acceleration_y, pose.acceleration_z, 
                pose.angular_rate_x, pose.angular_rate_y, pose.angular_rate_z, 
                pose.vx, pose.vy, pose.vz, 
                yaw_rad, yaw_deg, 
            ])
            
            # get future valid flag by checking the next future token
            fut_valid_flag = True

            # get the sample info
            # TODO: check the info format between nuplan and nuscenes
            current_index = lidar_pc_token_list.index(lidar_pc_token)
            prev_token = lidar_pc_token_list[current_index - 1]
            next_token = lidar_pc_token_list[current_index + 1]

            # map_location
            # if map_location == "us-ma-boston":
            #     map_location = "boston-seaport"
            if map_location == "las_vegas":
                map_location = "us-nv-las-vegas-strip"

            info = {
                "lidar_path": lidar_path,
                "token": uuid.uuid5(uuid.NAMESPACE_DNS, lidar_pc_token).hex,
                "prev": uuid.uuid5(uuid.NAMESPACE_DNS, prev_token).hex if image_f0.prev_token else None,
                "next": uuid.uuid5(uuid.NAMESPACE_DNS, next_token).hex if image_f0.next_token else None,
                "can_bus": can_bus,
                "frame_idx": frame_idx,  # temporal related info
                "sweeps": [],
                "cams": dict(),
                "scene_token": scene_token, 
                "lidar_pc": image_f0.lidar_pc.token, 
                "lidar2ego_translation": list(cs_record.translation_np), 
                "lidar2ego_rotation": list(cs_record.quaternion), 
                "ego2global_translation": list(pose_record.translation_np), 
                "ego2global_rotation": list(pose_record.quaternion), 
                "timestamp": image_f0.timestamp,
                "fut_valid_flag": fut_valid_flag,
                "map_location": map_location, 
                "roadblock_ids": roadblock_ids,
            }

            # 加入红绿灯信息和其他tag信息
            lidar_pc = nuplan_db.image[sample_token_buffer[lidar_pc_token][0]].lidar_pc
            if lidar_pc.traffic_lights != []:
                info['traffic_light'] = set([tl.status for tl in lidar_pc.traffic_lights])
            else:
                info['traffic_light'] = set()
            tags = set([tag.type for tag in lidar_pc.scenario_tags])
            # info["traffic_light"] = "traffic_light" in tags
            info["lidar_token"] = lidar_pc_token
            info['tags'] = tags
            info['scene_file'] = log_file_name

            l2e_r = cs_record.rotation
            l2e_t = cs_record.translation_np
            e2g_r = pose_record.quaternion
            e2g_t = pose_record.translation_np
            l2e_r_mat = Quaternion(l2e_r).rotation_matrix
            e2g_r_mat = e2g_r.rotation_matrix

            # Add Future e2g_r_mat and e2g_t
            fut_e2g_r_mats = []
            fut_e2g_ts = []
            for fut_lidar_pc in fut_lidar_pc_deq:
                fut_pose_record = nuplan_db.ego_pose.get(fut_lidar_pc.ego_pose_token)
                fut_e2g_r = fut_pose_record.quaternion
                fut_e2g_t = fut_pose_record.translation_np
                fut_e2g_r_mat = fut_e2g_r.rotation_matrix
                fut_e2g_r_mats.append(fut_e2g_r_mat)
                fut_e2g_ts.append(fut_e2g_t)

            # TODO: obtain 8 camera images' information per frame
            # refer to the nuscenes code: obtain_sensor2top
            lidar2ego_translation = cs_record.translation_np
            lidar2ego_rotation = cs_record.quaternion
            cams = {}
            for image_id in sample_token_buffer[lidar_pc_token]:
                image: Image = nuplan_db.image[image_id]
                camera: Camera = image.camera
                ego: EgoPose = image.ego_pose

                sensor2ego_rotation = Quaternion(camera.rotation)
                sensor2ego_translation = camera.translation
                lidar2ego_rotation_inv = lidar2ego_rotation.inverse
                # sensor to lidar rotation: lidar2ego_rotation_inv * sensor2ego_rotation
                sensor2lidar_rotation = lidar2ego_rotation_inv * sensor2ego_rotation
                # sensor to lidar translation
                sensor2lidar_translation = lidar2ego_rotation_inv.rotate(sensor2ego_translation - lidar2ego_translation)

                cam = {
                    'data_path': f"./data/nuplan/dataset/nuplan-v1.1/sensor_blobs/{image.filename}", 
                    'type': camera.channel, 
                    'sample_data_token':  image.token, # TODO: Not sure what's this
                    'sensor2ego_translation': np.array(sensor2ego_translation), 
                    'sensor2ego_rotation': sensor2ego_rotation.elements, 
                    'ego2global_translation': [ego.x, ego.y, ego.z], 
                    'ego2global_rotation': [ego.qw, ego.qx, ego.qy, ego.qz], 
                    'timestamp': image.timestamp, 
                    'sensor2lidar_rotation': sensor2lidar_rotation.rotation_matrix, 
                    'sensor2lidar_translation': np.array(sensor2lidar_translation.tolist()),
                    'cam_intrinsic': np.array(camera.intrinsic)
                }

                cams[camera.channel] = cam
        
            info["cams"] = cams
                
            # TODO: obtain sweeps for a single key-frame. for now, sweeps = []
            # refer to the nuscenes code: obtain_sensor2top
            sweeps = []
            info['sweeps'] = sweeps

            ####################################################
            #################  Get env agents  #################
            ####################################################
            boxes = image_f0.lidar_boxes
            # get ego yaw in global frame
            def quaternion_to_yaw(q):
                """
                Convert a quaternion to a yaw angle (rotation around Z-axis).
                q: array-like of shape (4,), representing [w, x, y, z]
                """
                w, x, y, z = q
                # Compute yaw angle
                siny_cosp = 2 * (w * z + x * y)
                cosy_cosp = 1 - 2 * (y**2 + z**2)
                yaw = np.arctan2(siny_cosp, cosy_cosp)
                return yaw
            
            global_yaw = quaternion_to_yaw(info["ego2global_rotation"])
            if boxes:
                # the trajs and vel info can generate by nuplan devkit code
                tracks = [
                    nuplan_db.get('track', box.track_token)
                    for box in boxes
                ]
                locs = [np.dot(np.transpose(e2g_r_mat), box.translation - e2g_t) for box in boxes]
                dims = np.array([box.size for box in boxes]).reshape(-1, 3)
                rots = np.array([box.yaw for box in boxes]).reshape(-1, 1)

                velocity = np.array([[box.vx, box.vy] for box in boxes])
                # TODO: no valid_flag in nuplan，set to all True for now 
                valid_flag = np.array([True for ox in boxes], dtype=bool).reshape(-1)
                # convert velo from global to lidar
                for i in range(len(boxes)):
                    velo = np.array([*velocity[i], 0.0])
                    velo = velo @ np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(
                        l2e_r_mat).T
                    velocity[i] = velo[:2]

                names = [b.category.name for b in boxes]
                # we need to convert rot to SECOND format.

                # rots from global yaw to ego yaw:
                rots = -rots + global_yaw
                # gt_boxes = np.concatenate([locs, dims, -rots - np.pi / 2], axis=1)
                gt_boxes = np.concatenate([locs, dims, rots], axis=1)
                # get future coords for each box
                num_box = len(boxes)
                gt_fut_trajs = np.zeros((num_box, fut_ts, 2))
                gt_fut_yaw = np.zeros((num_box, fut_ts))
                gt_fut_masks = np.zeros((num_box, fut_ts))
                # gt_boxes_yaw = -(gt_boxes[:,6] + np.pi / 2) #TODO: -rots + global_yaw
                gt_boxes_yaw = -(gt_boxes[:,6]) #TODO: -rots + global_yaw
                # agent lcf feat (x, y, yaw, vx, vy, width, length, height, type)
                agent_lcf_feat = np.zeros((num_box, 9))
                gt_fut_goal = np.zeros((num_box))

                # Precompute transformations
                ego_translation = np.array([pose_record.x, pose_record.y, pose_record.z])
                ego_rotation = Quaternion(
                    w=pose_record.qw, x=pose_record.qx, y=pose_record.qy, z=pose_record.qz
                )
                cs_translation = np.array(cs_record.translation)
                cs_rotation = Quaternion(cs_record.rotation)

                # Compute the inverse transformation matrices once
                T_ego_inv = transform_matrix(ego_translation, ego_rotation, inverse=True)
                T_cs_inv = transform_matrix(cs_translation, cs_rotation, inverse=True)
                T_total = np.dot(T_cs_inv, T_ego_inv)

                # Precompute inverse yaw rotations
                ego_rotation_inv_yaw = -ego_rotation.yaw_pitch_roll[0]
                cs_rotation_inv_yaw = -cs_rotation.yaw_pitch_roll[0]
                total_inv_yaw = ego_rotation_inv_yaw + cs_rotation_inv_yaw

                for idx, track in enumerate(tracks):
                    cur_box = boxes[idx]
                    agent_lcf_feat[idx, 0:2] = locs[idx][:2]
                    agent_lcf_feat[idx, 2] = gt_boxes_yaw[idx]
                    agent_lcf_feat[idx, 3:5] = velocity[idx]
                    agent_lcf_feat[idx, 5:8] = (track.width, track.length, track.height)
                    agent_lcf_feat[idx, 8] = cat2idx.get(track.category.name, -1)

                    box_next = cur_box
                    for j in range(fut_ts):
                        # Move forward certain steps to match frame rate
                        steps = 0
                        while steps < (FRAME_RATE_IMAGE // TARGET_FRAME_RATE) and box_next:
                            box_next = box_next.next
                            steps += 1
                        if not box_next:
                            gt_fut_trajs[idx, j:] = 0
                            break
                        # Move box to ego vehicle coord system.
                        # box_next.translate(-fut_e2g_ts[j])
                        # box_next.rotate(Quaternion(fut_e2g_r_mats[j]).inverse)

                        # transform the global coor to ego coor
                        global_to_ego = Quaternion(matrix=fut_e2g_r_mats[j]).inverse
                        local_translation = global_to_ego.rotate(box_next.translation_np - fut_e2g_ts[j])

                        # Calculate relative displacement to the current frame
                        cur_local_translation = global_to_ego.rotate(cur_box.translation_np - fut_e2g_ts[j])
                        gt_fut_trajs[idx, j] = local_translation[:2] - cur_local_translation[:2]
                        gt_fut_masks[idx, j] = 1

                        # calc yaw diff
                        global_yaw = box_next.yaw
                        ego_yaw = quaternion_to_yaw(Quaternion(matrix=fut_e2g_r_mats[j]))
                        local_yaw = global_yaw - ego_yaw
                        
                        cur_global_yaw = cur_box.yaw
                        cur_local_yaw = cur_global_yaw - ego_yaw
                        
                        gt_fut_yaw[idx, j] = local_yaw - cur_local_yaw
                        
                        cur_box = box_next

                    # Get agent goal
                    gt_fut_coords = np.cumsum(gt_fut_trajs[idx], axis=-2)
                    coord_diff = gt_fut_coords[-1] - gt_fut_coords[0]
                    if np.abs(coord_diff).max() < 1.0:  # Static agent
                        gt_fut_goal[idx] = 9
                    else:
                        box_mot_yaw = np.arctan2(coord_diff[1], coord_diff[0]) + np.pi
                        gt_fut_goal[idx] = int(box_mot_yaw // (np.pi / 4))  # Goal direction class 0-8

            # handel no boxes case
            else:
                gt_boxes = np.array([])
                names = []
                velocity = np.array([])
                valid_flag = np.array([], dtype=bool)
                gt_fut_trajs = np.array([])
                gt_fut_masks = np.array([])
                agent_lcf_feat = np.array([])
                gt_fut_yaw = np.array([])
                gt_fut_goal = np.array([])
                print(f"\nWARNING: No boxes in {lidar_pc_token}!")


            info['gt_boxes'] = gt_boxes
            info['gt_names'] = np.array(names)
            info['gt_velocity'] = velocity.reshape(-1, 2)
            info['num_lidar_pts'] = np.array(
                [1 for _ in boxes])
            info['num_radar_pts'] = np.array(
                [1 for _ in boxes]) 
            info['valid_flag'] = valid_flag
            info['gt_agent_fut_trajs'] = gt_fut_trajs.reshape(-1, fut_ts*2).astype(np.float32)
            info['gt_agent_fut_masks'] = gt_fut_masks.reshape(-1, fut_ts).astype(np.float32)
            info['gt_agent_lcf_feat'] = agent_lcf_feat.astype(np.float32)
            info['gt_agent_fut_yaw'] = gt_fut_yaw.astype(np.float32)
            info['gt_agent_fut_goal'] = gt_fut_goal.astype(np.float32)

            ####################################################
            #################  Get Ego agents  #################
            ####################################################
            
            # get ego history traj (off format)
            ego_his_trajs = np.zeros((his_ts + 1, 3))
            ego_his_trajs_diff = np.zeros((his_ts + 1, 3))
            for j, lidar_pc in enumerate(reversed(lidar_pc_deq)):
                i = his_ts - j
                pose_mat = get_global_sensor_pose(lidar_pc.token, nuplan_db, inverse=False)
                ego_his_trajs[i] = pose_mat[:3, 3]
                
                if i < his_ts:  # 不计算最后一帧的差值
                    next_lidar_pc = lidar_pc_deq[-i-1] if i > 0 else cur_lidar_pc
                    pose_mat_next = get_global_sensor_pose(next_lidar_pc.token, nuplan_db, inverse=False)
                    ego_his_trajs_diff[i] = pose_mat_next[:3, 3] - ego_his_trajs[i]

            # global to ego at lcf
            ego_his_trajs = ego_his_trajs - np.array([pose_record.x, pose_record.y, pose_record.z])
            rot_mat = Quaternion([pose_record.qw, pose_record.qx, pose_record.qy, pose_record.qz]).inverse.rotation_matrix
            ego_his_trajs = np.dot(rot_mat, ego_his_trajs.T).T
            # ego to lidar at lcf
            ego_his_trajs = ego_his_trajs - np.array(cs_record.translation)
            rot_mat = Quaternion(cs_record.rotation).inverse.rotation_matrix
            ego_his_trajs = np.dot(rot_mat, ego_his_trajs.T).T
            ego_his_trajs = ego_his_trajs[1:] - ego_his_trajs[:-1]
            
            # get ego futute traj (offset format)
            ego_fut_trajs = np.zeros((fut_ts + 1, 3))
            ego_fut_masks = np.ones((fut_ts + 1))
            for i, lidar_pc in enumerate(fut_lidar_pc_deq):
                pose_mat = get_global_sensor_pose(lidar_pc.token, nuplan_db, inverse=False)
                ego_fut_trajs[i] = pose_mat[:3, 3]

            # global to ego at lcf
            ego_fut_trajs = ego_fut_trajs - np.array([pose_record.x, pose_record.y, pose_record.z])
            rot_mat = Quaternion([pose_record.qw, pose_record.qx, pose_record.qy, pose_record.qz]).inverse.rotation_matrix
            ego_fut_trajs = np.dot(rot_mat, ego_fut_trajs.T).T
            # ego to lidar at lcf
            ego_fut_trajs = ego_fut_trajs - np.array(cs_record.translation)
            rot_mat = Quaternion(cs_record.rotation).inverse.rotation_matrix
            ego_fut_trajs = np.dot(rot_mat, ego_fut_trajs.T).T           
            # drive command according to final fut step offset from lcf
            if ego_fut_trajs[-1][1] <= -2:
                command = np.array([1, 0, 0])  # Turn Right
            elif ego_fut_trajs[-1][1] >= 2:
                command = np.array([0, 1, 0])  # Turn Left
            else:
                command = np.array([0, 0, 1])  # Go Straight

            # offset from lcf -> per-step offset
            ego_fut_trajs = ego_fut_trajs[1:] - ego_fut_trajs[:-1]

            ### ego lcf feat (vx, vy, ax, ay, w, length, width, vel, steer), w: yaw角速度
            ego_lcf_feat = np.zeros(9)

            # 计算v0和路径曲率Kappa
            v0 = np.sqrt(pose_record.vx**2 + pose_record.vy**2)
            w = pose_record.angular_rate_z
            if abs(v0) > 1e-6:  # 避免除以零
                Kappa = w / v0
            else:
                Kappa = 0  # 当速度接近零时,设置 kappa 为 0
            
            ego_lcf_feat[:2] = np.array([pose_record.vx, pose_record.vy])
            ego_lcf_feat[2:4] = np.array([pose_record.acceleration_x, pose_record.acceleration_y])
            ego_lcf_feat[4] = pose_record.angular_rate_z
            ego_lcf_feat[5:7] = np.array([ego_length, ego_width])
            ego_lcf_feat[7] = v0
            ego_lcf_feat[8] = Kappa

            info['gt_ego_his_trajs'] = ego_his_trajs[:, :2].astype(np.float32)
            info['gt_ego_fut_trajs'] = ego_fut_trajs[:, :2].astype(np.float32)
            info['gt_ego_fut_masks'] = ego_fut_masks[1:].astype(np.float32)
            info['gt_ego_fut_cmd'] = command.astype(np.float32)
            info['gt_ego_lcf_feat'] = ego_lcf_feat.astype(np.float32)

            nuplan_infos.append(info)
        

        # output .pkl file
        import pickle
        with open(output_pkl_path, 'wb') as file:
            pickle.dump(nuplan_infos, file)
            print(f"Save {output_pkl_path} successfully!")

    if use_multiprocessing:
        print("Using multiprocessing to process the logs...")
        import joblib
        import multiprocessing
        available_cpu = multiprocessing.cpu_count()
        print(f"Available CPU: {available_cpu}")
        jobs_needed = len(splits)
        used_cpu = max(1, min(available_cpu - 1, jobs_needed))
        joblib.Parallel(n_jobs=used_cpu)(
            joblib.delayed(process_log)(log_file_name, output_dir, args.location)
            for log_file_name in tqdm(splits)
        )
    else:
        print("Using single process to process the logs...")
        for log_file_name in tqdm(splits):
            process_log(log_file_name, output_dir, args.location)

    return None, None

def get_global_sensor_pose(lidar_pc_token, nuplan_db, inverse=False):
    lidar_pc = nuplan_db.get('lidar_pc', lidar_pc_token)

    sd_ep = lidar_pc.ego_pose
    sd_ld = lidar_pc.lidar
    if inverse is False:
        global_from_ego = transform_matrix(
            [sd_ep.x, sd_ep.y, sd_ep.z],
            Quaternion(sd_ep.qw, sd_ep.qx, sd_ep.qy, sd_ep.qz),
            inverse=False
        )
        ego_from_sensor = transform_matrix(
            sd_ld.translation,
            Quaternion(sd_ld.rotation),
            inverse=False
        )
        pose = global_from_ego.dot(ego_from_sensor)
    else:
        global_from_ego = transform_matrix(
            [sd_ep.x, sd_ep.y, sd_ep.z],
            Quaternion(sd_ep.qw, sd_ep.qx, sd_ep.qy, sd_ep.qz),
            inverse=True
        )
        ego_from_sensor = transform_matrix(
            sd_ld.translation,
            Quaternion(sd_ld.rotation),
            inverse=True
        )
        pose = global_from_ego.dot(ego_from_sensor)
    return pose


parser = argparse.ArgumentParser(description="Data converter arg parser")
parser.add_argument(
    "--root_path",
    type=str,
    default="./data/kitti",
    help="specify the root path of dataset",
)
parser.add_argument(
    "--version",
    type=str,
    default="v1.0",
    required=False,
    help="specify the dataset version, no need for kitti",
)

parser.add_argument(
    "--out_dir",
    type=str,
    required=False,
    help="name of info pkl",
)
parser.add_argument(
    "--workers", type=int, default=None, help="number of threads to be used"
)
parser.add_argument(
    '--use_multiprocessing',
    type=lambda x: bool(strtobool(x)),
    default=True,
    help='Use multiprocessing (default: True)'
)

parser.add_argument(
    '--overwrite',
    type=lambda x: bool(strtobool(x)),
    default=True,
    help='Overwrite the existing file (default: True)'
)

parser.add_argument(
    "--location",
    default=None,
    help="Location of map",
)
args = parser.parse_args()


if __name__ == "__main__":
    if args.version == "trainval":
        create_nuplan_infos(
            root_path=args.root_path,
            out_path=args.out_dir,
            version="trainval",
            use_multiprocessing=args.use_multiprocessing,
        )
    elif args.version == "mini":
        create_nuplan_infos(
            root_path=args.root_path,
            out_path=args.out_dir,
            version='mini',
            use_multiprocessing=args.use_multiprocessing,
        )
    elif args.version == "test":
        create_nuplan_infos(
            root_path=args.root_path,
            out_path=args.out_dir,
            version="test",
            use_multiprocessing=args.use_multiprocessing,
        )
    else:
        # try:
        create_nuplan_infos(
        root_path=args.root_path,
        out_path=args.out_dir,
        version=args.version,
        use_multiprocessing=args.use_multiprocessing,
        )
        # except Exception as e:
        #     raise ValueError(f"Dataset not supported version: {args.version}")

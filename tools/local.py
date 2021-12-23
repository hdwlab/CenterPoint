from tqdm import tqdm
import open3d as o3d
import glob
import single_infernece as si
from det3d.core.input.voxel_generator import VoxelGenerator
from ros_node.utils import make_message_bounding_box_array, \
    make_message_lidar_detected_object_array, \
    remove_points_in_boxes
import sensor_msgs
import ros_numpy
import rosbag
from torch.nn.parallel import DistributedDataParallel
from det3d.torchie.trainer.utils import all_gather, synchronize
from det3d.torchie.trainer import get_dist_info, load_checkpoint
from det3d.torchie.apis import (
    batch_processor,
    build_optimizer,
    get_root_logger,
    init_dist,
    set_random_seed,
    train_detector,
)
from det3d.torchie import Config
from det3d.models import build_detector
from det3d.datasets import build_dataloader, build_dataset
from det3d import __version__, torchie
import yaml
import torch
import numpy as np
import os
import json
import copy
import argparse
import sys
sys.path.append('CenterPoint/det3d')
try:
    import apex
except:
    print("No APEX!")


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a detector")
    parser.add_argument("config", help="train config file path")
    parser.add_argument("--work_dir", help="the dir to save logs and models")
    parser.add_argument(
        "--checkpoint", help="the dir to checkpoint which the model read from"
    )
    parser.add_argument(
        "--txt_result",
        type=bool,
        default=False,
        help="whether to save results to standard KITTI format of txt type",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=1,
        help="number of gpus to use " "(only applicable to non-distributed training)",
    )
    parser.add_argument(
        "--launcher",
        choices=["none", "pytorch", "slurm", "mpi"],
        default="none",
        help="job launcher",
    )
    parser.add_argument("--speed_test", action="store_true")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--testset", action="store_true")
    parser.add_argument("--filelist", type=str, default=None,
                        help="filename of filelist of bin or pcd or rosbag files. "
                             "you need to specify either --filelist of --file.")
    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)

    return args


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    cfg.local_rank = args.local_rank

    # update configs according to CLI args
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    distributed = False
    if "WORLD_SIZE" in os.environ:
        distributed = int(os.environ["WORLD_SIZE"]) > 1

    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://")
        cfg.gpus = torch.distributed.get_world_size()
    else:
        cfg.gpus = args.gpus

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")

    # init logger before other steps
    logger = get_root_logger(cfg.log_level)
    logger.info("Distributed testing: {}".format(distributed))
    logger.info(
        f"torch.backends.cudnn.benchmark: {torch.backends.cudnn.benchmark}")

    # define model
    voxel_generator = VoxelGenerator(
        voxel_size=cfg.voxel_generator.voxel_size,
        point_cloud_range=cfg.voxel_generator.range,
        max_num_points=cfg.voxel_generator.max_points_in_voxel,
        max_voxels=cfg.voxel_generator.max_voxel_num,
    )
    model = build_detector(
        cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    checkpoint = load_checkpoint(
        model, args.checkpoint, map_location="cpu")
    model = model.cuda()
    model.eval()
    torch.cuda.synchronize()
    args.filelist = './scenes/scene_1'
    args.json_output_file = './detection/scene_1/pred.json'
    if args.filelist:
        #filelist = glob.glob(args.filelist+'*')
        filelist = glob.glob(args.filelist)
        filelist = sorted(filelist)
        detections = {}
        for filename in tqdm(filelist):
            pcd_files = glob.glob(f"{filename}/Images/*.pcd")
            pcd_files = sorted(pcd_files)
            for pcd_file in pcd_files:
                points = np.array(o3d.io.read_point_cloud(pcd_file).points)
                # convert size for centerpoint
                pointcloud = np.zeros(
                    shape=(points.shape[0], 5), dtype=np.float64)
                pointcloud[:, :3] = points
                # convert table
                # -----------------------------
                # | nuscenes  |  METI(rosbag) |
                # -----------------------------
                # |     x     |      -y       |
                # |     y     |       x       |
                # |     z     |       z       |
                # |   length  |    width_3d   |
                # |   width   |   height_3d   |
                # |   height  |   length_3d   |
                # |    yaw    |   rotation_y  |
                # -----------------------------
                # convert pointclouds(ros->nuscenes)
                pointcloud = pointcloud[:, [1, 0, 2, 3, 4]]
                pointcloud[:, 0] = pointcloud[:, 0]*(-1)
                pointcloud[:, 4] = 0
                voxels, coords, num_points = voxel_generator.generate(
                    pointcloud)
                num_voxels = np.array([voxels.shape[0]], dtype=np.int64)
                grid_size = voxel_generator.grid_size
                coords = np.pad(coords, ((0, 0), (1, 0)),
                                mode='constant', constant_values=0)
                voxels = torch.tensor(
                    voxels, dtype=torch.float32, device=device).cuda()
                coords = torch.tensor(
                    coords, dtype=torch.int32, device=device).cuda()
                num_points = torch.tensor(
                    num_points, dtype=torch.int32, device=device).cuda()
                num_voxels = torch.tensor(
                    num_voxels, dtype=torch.int32, device=device).cuda()
                inputs = dict(
                    voxels=voxels,
                    num_points=num_points,
                    num_voxels=num_voxels,
                    coordinates=coords,
                    shape=[grid_size]
                )

                # inference
                with torch.no_grad():
                    outputs = model(inputs, return_loss=False)
                prediction_frame = outputs[0]
                detections_in_frame = []
                class_names = ['car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                               'barrier', 'motorcycle', 'bicycle', "pedestrian", "traffic_cone"]
                # nuscenes boxes ([N, 9] Tensor): normal boxes: x, y, z, w, l, h, vx, vy,i
                # convert table
                # -----------------------------
                # | nuscenes  |  METI(rosbag) |
                # -----------------------------
                # |     x     |      -y       |
                # |     y     |       x       |
                # |     z     |       z       |
                # |   length  |    width_3d   |
                # |   width   |   height_3d   |
                # |   height  |   length_3d   |
                # |    yaw    |   rotation_y  |
                # -----------------------------
                coordinate_idxes = {'z': 2, 'y': 0, 'x': 1,
                                    'width': 3,  'length': 4, 'height': 5, 'rotation': -1}
                # global_object_count = 0
                for object_id, (score, label, box_detection) in enumerate(zip(prediction_frame['scores'], prediction_frame['label_preds'], prediction_frame['box3d_lidar'])):
                    class_name = class_names[int(label.cpu().numpy())]
                    dict_per_obj = {}
                    dict_per_obj['score'] = score.to(
                        'cpu').detach().numpy().astype(np.float64)
                    dict_per_obj['class'] = class_name
                    for name, idx in coordinate_idxes.items():
                        dict_per_obj[name] = box_detection[idx].to(
                            'cpu').detach().numpy().astype(np.float64)
                        if name == 'y':
                            dict_per_obj[name] *= (-1.0)
                        elif name == 'rotation':
                            dict_per_obj[name] /= 180.0
                    detections_in_frame.append(dict_per_obj)
                file_id = pcd_file.split("/")[-1].split(".")[0].split('_')[1]
                detections.update({file_id: detections_in_frame})
        if not os.path.isdir(os.path.dirname(args.json_output_file)):
            os.makedirs(os.path.dirname(args.json_output_file))
        with open(args.json_output_file, 'w') as f:
            json.dump(detections, f, cls=MyEncoder)
    return

    # create rosbag
    outbag = rosbag.Bag('out.bag', "w")
    with rosbag.Bag("sample.bag", "r") as inbag:
        for topic, msg, t in inbag.read_messages():
            outbag.write(topic, msg, t)
            if not 'PointCloud2' in type(msg).__name__:
                continue
            print(t)
            # convert PointCloud2 to ndarray
            msg.__class__ = sensor_msgs.msg._PointCloud2.PointCloud2
            points = ros_numpy.numpify(
                msg)[["x", "y", "z", "intensity"]]
            pointcloud = np.array(points.tolist())

            # create voxel
            msg_cloud = ros_numpy.point_cloud2.pointcloud2_to_array(msg)
            np_p = si.get_xyz_points(msg_cloud, True).reshape([-1, 5])

            # convert table
            # -----------------------------
            # | nuscenes  |  METI(rosbag) |
            # -----------------------------
            # |     x     |      -y       |
            # |     y     |       x       |
            # |     z     |       z       |
            # |   length  |    width_3d   |
            # |   width   |   height_3d   |
            # |   height  |   length_3d   |
            # |    yaw    |   rotation_y  |
            # -----------------------------
            # convert pointclouds(ros->nuscenes)
            np_p[:, 0], np_p[:, 1] = np_p[:, 1]*(-1.0), np_p[:, 0]
            np_p[:, 4] = 0
            voxels, coords, num_points = voxel_generator.generate(np_p)
            num_voxels = np.array([voxels.shape[0]], dtype=np.int64)
            grid_size = voxel_generator.grid_size
            coords = np.pad(coords, ((0, 0), (1, 0)),
                            mode='constant', constant_values=0)
            voxels = torch.tensor(
                voxels, dtype=torch.float32, device=device).cuda()
            coords = torch.tensor(
                coords, dtype=torch.int32, device=device).cuda()
            num_points = torch.tensor(
                num_points, dtype=torch.int32, device=device).cuda()
            num_voxels = torch.tensor(
                num_voxels, dtype=torch.int32, device=device).cuda()
            inputs = dict(
                voxels=voxels,
                num_points=num_points,
                num_voxels=num_voxels,
                coordinates=coords,
                shape=[grid_size]
            )

            # inference
            with torch.no_grad():
                outputs = model(inputs, return_loss=False)
            prediction_frame = outputs[0]
            detections_in_frame = []
            class_names = ['car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                           'barrier', 'motorcycle', 'bicycle', "pedestrian", "traffic_cone"]
            for label, score, box_detection in zip(prediction_frame['label_preds'], prediction_frame['scores'], prediction_frame['box3d_lidar']):
                detection_per_object = {}
                detection_per_object['score'] = score.to(
                    'cpu').detach().numpy().astype(np.float64)
                detection_per_object['class'] = str(class_names[label.to(
                    'cpu').detach().numpy().astype(np.int32)])
                # nuscenes boxes ([N, 9] Tensor): normal boxes: x, y, z, w, l, h, vx, vy,i
                detection_per_object['x'] = box_detection[0].to(
                    'cpu').detach().numpy().astype(np.float64)
                detection_per_object['y'] = box_detection[1].to(
                    'cpu').detach().numpy().astype(np.float64)
                detection_per_object['z'] = box_detection[2].to(
                    'cpu').detach().numpy().astype(np.float64)
                detection_per_object['width'] = box_detection[3].to(
                    'cpu').detach().numpy().astype(np.float64)
                detection_per_object['length'] = box_detection[4].to(
                    'cpu').detach().numpy().astype(np.float64)
                detection_per_object['height'] = box_detection[5].to(
                    'cpu').detach().numpy().astype(np.float64)
                detection_per_object['rotation'] = box_detection[-1].to(
                    'cpu').detach().numpy().astype(np.float64)
                detections_in_frame.append(detection_per_object)

            header = msg.header
            bbs = [[box['class'],
                    box['x'],
                    box['y'],
                    box['z'],
                    box['height'],
                    box['width'],
                    box['length'],
                    box['rotation'],
                    box['score']] for box in detections_in_frame]
            # to rosbag
            msg_bbox = make_message_bounding_box_array(bbs, header=header)
            msg_detection = make_message_lidar_detected_object_array(bbs,
                                                                     header=header,
                                                                     version='0.3.0')
            points_without_objects = remove_points_in_boxes(
                pointcloud, bbs,
                score_threshold=0.2,
                margin=0.5
            )
            new_points = np.core.records.fromarrays(points_without_objects.transpose(),
                                                    names='x, y, z, intensity',
                                                    formats='f, f, f, f')
            msg_new_points = ros_numpy.msgify(
                sensor_msgs.msg.PointCloud2, new_points)
            msg_new_points.header = msg.header
            outbag.write('/bounding_box', msg_bbox, t)
            outbag.write('/detected_objects', msg_detection, t)
            outbag.write('/points_without_objects', msg_new_points, t)


if __name__ == "__main__":
    main()

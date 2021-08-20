import argparse
import copy
import json
import os
import sys

try:
    import apex
except:
    print("No APEX!")
import numpy as np
import torch
import yaml
from det3d import __version__, torchie
from det3d.datasets import build_dataloader, build_dataset
from det3d.models import build_detector
from det3d.torchie import Config
from det3d.torchie.apis import (
    batch_processor,
    build_optimizer,
    get_root_logger,
    init_dist,
    set_random_seed,
    train_detector,
)
from det3d.torchie.trainer import get_dist_info, load_checkpoint
from det3d.torchie.trainer.utils import all_gather, synchronize
from torch.nn.parallel import DistributedDataParallel
import pickle
import time


def save_pred(pred, root):
    with open(os.path.join(root, "prediction.pkl"), "wb") as f:
        pickle.dump(pred, f)


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

    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)

    return args


def main():

    # torch.manual_seed(0)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # np.random.seed(0)

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

    # init logger before other steps
    logger = get_root_logger(cfg.log_level)
    logger.info("Distributed testing: {}".format(distributed))
    logger.info(
        f"torch.backends.cudnn.benchmark: {torch.backends.cudnn.benchmark}")

    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)

    # print(all_predictions)

    # predictions = {}
    # for p in all_predictions:
    #     predictions.update(p)

    # for key in sorted(predictions.keys()):
    #     detections_in_frame = []
    #     prediction_frame = predictions[key]
    #     for label, score, box_detection in zip(prediction_frame['label_preds'], prediction_frame['scores'], prediction_frame['box3d_lidar']):
    #         detection_per_object = {}
    #         detection_per_object['score'] = score.to(
    #             'cpu').detach().numpy().astype(np.float64)
    #         detection_per_object['class'] = str(label.to(
    #             'cpu').detach().numpy().astype(np.int32))
    #         # boxes ([N, 9] Tensor): normal boxes: x, y, z, w, l, h, vx, vy,
    #         # convert table
    #         # -----------------------------
    #         # | nuscenes  |     METI      |
    #         # -----------------------------
    #         # |     x     |      -y       |
    #         # |     y     |       x       |
    #         # |     z     |       z       |
    #         # |   length  |    width_3d   |
    #         # |   width   |   height_3d   |
    #         # |   height  |   length_3d   |
    #         # |    yaw    |   rotation_y  |
    #         # -----------------------------
    #         detection_per_object['x'] = - \
    #             box_detection[1].to('cpu').detach().numpy().astype(np.float64)
    #         detection_per_object['y'] = box_detection[0].to(
    #             'cpu').detach().numpy().astype(np.float64)
    #         detection_per_object['z'] = box_detection[2].to(
    #             'cpu').detach().numpy().astype(np.float64)
    #         detection_per_object['length'] = box_detection[3].to(
    #             'cpu').detach().numpy().astype(np.float64)
    #         detection_per_object['width'] = box_detection[5].to(
    #             'cpu').detach().numpy().astype(np.float64)
    #         detection_per_object['height'] = box_detection[4].to(
    #             'cpu').detach().numpy().astype(np.float64)
    #         detection_per_object['rotation'] = box_detection[-1].to(
    #             'cpu').detach().numpy().astype(np.float64)
    #         detections_in_frame.append(detection_per_object)
    # import pdb
    # pdb.set_trace()
    import rosbag
    import ros_numpy
    import sensor_msgs
    from ros_node.utils import make_message_bounding_box_array, \
        make_message_lidar_detected_object_array, \
        remove_points_in_boxes
    out_size_factor = 2
    pc_range = [-54, -54]
    voxel_size = [0.075, 0.075]
    #w, l = w / voxel_size[0] / self.out_size_factor, l / voxel_size[1] / self.out_size_factor
    # coor_x, coor_y = (x - pc_range[0]) / voxel_size[0] / self.out_size_factor, \
    #(y - pc_range[1]) / voxel_size[1] / self.out_size_factor
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
            import single_infernece as si
            msg_cloud = ros_numpy.point_cloud2.pointcloud2_to_array(msg)

            np_p = si.get_xyz_points(msg_cloud, True).reshape([-1, 5])
            np_p[:, 4] = 0

            pointcloud = np.array(points.tolist())
            # print(pointcloud[0])
            # #pointcloud[:, 3] /= 255.0
            # pointcloud_centerpoint = []
            # for i, arr in enumerate(pointcloud):
            #     if i % 3 == 0:
            #         # import pdb
            #         # pdb.set_trace()
            #         # pointcloud_centerpoint.append(
            #         #     [0.0]+[-arr[1], arr[0], arr[2], arr[-1]]+[0.0])
            #         coor_x, coor_y = (arr[0] - pc_range[0]) / voxel_size[0] / out_size_factor, \
            #             (arr[1] - pc_range[1]) / \
            #             voxel_size[1] / out_size_factor
            #         # pointcloud_centerpoint.append(
            #         #     [0.0]+[coor_x, coor_y, arr[2], arr[3]]+[0.0])
            #         pointcloud_centerpoint.append(
            #             [coor_x, coor_y, arr[2]]+[0.0, 0.0])

            from det3d.core.input.voxel_generator import VoxelGenerator
            voxel_generator = VoxelGenerator(
                voxel_size=cfg.voxel_generator.voxel_size,
                point_cloud_range=cfg.voxel_generator.range,
                max_num_points=cfg.voxel_generator.max_points_in_voxel,
                max_voxels=cfg.voxel_generator.max_voxel_num,
            )
            voxels, coords, num_points = voxel_generator.generate(np_p)

            num_voxels = np.array([voxels.shape[0]], dtype=np.int64)
            grid_size = voxel_generator.grid_size
            coords = np.pad(coords, ((0, 0), (1, 0)),
                            mode='constant', constant_values=0)
            device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")
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

            # import pdb
            # pdb.set_trace()
            # print(pointcloud_centerpoint)
            # data_batch['points'] = torch.Tensor(
            #     np.array(pointcloud_centerpoint))
            model = build_detector(
                cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
            checkpoint = load_checkpoint(
                model, args.checkpoint, map_location="cpu")
            model = model.cuda()
            model.eval()
            torch.cuda.synchronize()
            print(inputs)
            with torch.no_grad():
                outputs = model(inputs, return_loss=False)
            prediction_frame = outputs[0]
            print(prediction_frame['scores'])
            detections_in_frame = []
            class_names = ['car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                           'barrier', 'motorcycle', 'bicycle', "pedestrian", "traffic_cone"]
            for label, score, box_detection in zip(prediction_frame['label_preds'], prediction_frame['scores'], prediction_frame['box3d_lidar']):
                detection_per_object = {}
                detection_per_object['score'] = score.to(
                    'cpu').detach().numpy().astype(np.float64)
                detection_per_object['class'] = str(class_names[label.to(
                    'cpu').detach().numpy().astype(np.int32)])
                # boxes ([N, 9] Tensor): normal boxes: x, y, z, w, l, h, vx, vy,
                # convert table
                # -----------------------------
                # | nuscenes  |     METI      |
                # -----------------------------
                # |     x     |      -y       |
                # |     y     |       x       |
                # |     z     |       z       |
                # |   length  |    width_3d   |
                # |   width   |   height_3d   |
                # |   height  |   length_3d   |
                # |    yaw    |   rotation_y  |
                # -----------------------------
                # detection_per_object['x'] = box_detection[1].to(
                #     'cpu').detach().numpy().astype(np.float64)
                # detection_per_object['y'] = box_detection[0].to(
                #     'cpu').detach().numpy().astype(np.float64)*(-1.0)
                # detection_per_object['z'] = box_detection[2].to(
                #     'cpu').detach().numpy().astype(np.float64)
                # detection_per_object['height'] = box_detection[3].to(
                #     'cpu').detach().numpy().astype(np.float64)
                # detection_per_object['length'] = box_detection[5].to(
                #     'cpu').detach().numpy().astype(np.float64)
                # detection_per_object['width'] = box_detection[4].to(
                #     'cpu').detach().numpy().astype(np.float64)

                # detection_per_object['rotation'] = box_detection[-1].to(
                #     'cpu').detach().numpy().astype(np.float64)
                # detections_in_frame.append(detection_per_object)
                detection_per_object['x'] = box_detection[0].to(
                    'cpu').detach().numpy().astype(np.float64)
                detection_per_object['y'] = box_detection[1].to(
                    'cpu').detach().numpy().astype(np.float64)
                detection_per_object['z'] = box_detection[2].to(
                    'cpu').detach().numpy().astype(np.float64)
                detection_per_object['height'] = box_detection[5].to(
                    'cpu').detach().numpy().astype(np.float64)
                detection_per_object['length'] = box_detection[3].to(
                    'cpu').detach().numpy().astype(np.float64)
                detection_per_object['width'] = box_detection[4].to(
                    'cpu').detach().numpy().astype(np.float64)

                detection_per_object['rotation'] = box_detection[-1].to(
                    'cpu').detach().numpy().astype(np.float64)
                detections_in_frame.append(detection_per_object)

            header = msg.header
            # ここは検出結果がこのように抽出できるよう、CenterPoint用に作り変える必要がある
            bbs = [[box['class'],
                    box['x'],
                    box['y'],
                    box['z'],
                    box['height'],
                    box['width'],
                    box['length'],
                    box['rotation'],
                    box['score']] for box in detections_in_frame]
            msg_bbox = make_message_bounding_box_array(bbs, header=header)
            msg_detection = make_message_lidar_detected_object_array(bbs,
                                                                     header=header,
                                                                     version='0.3.0')

            points_without_objects = remove_points_in_boxes(
                pointcloud, bbs,
                score_threshold=0.2,
                margin=0.5
            )
            #points_without_objects[:, 3] *= 255.0
            new_points = np.core.records.fromarrays(points_without_objects.transpose(),
                                                    names='x, y, z, intensity',
                                                    formats='f, f, f, f')
            msg_new_points = ros_numpy.msgify(
                sensor_msgs.msg.PointCloud2, new_points)
            msg_new_points.header = msg.header
            # import pdb
            # pdb.set_trace()
            outbag.write('/bounding_box', msg_bbox, t)
            outbag.write('/detected_objects', msg_detection, t)
            outbag.write('/points_without_objects', msg_new_points, t)

    if args.local_rank != 0:
        return

    if not os.path.exists(args.work_dir):
        os.makedirs(args.work_dir)

    save_pred(predictions, args.work_dir)
    with open(os.path.join(args.work_dir, 'prediction.pkl'), 'rb') as f:
        predictions = pickle.load(f)

    result_dict, _ = dataset.evaluation(copy.deepcopy(
        predictions), output_dir=args.work_dir, testset=args.testset)

    if result_dict is not None:
        for k, v in result_dict["results"].items():
            print(f"Evaluation {k}: {v}")

    if args.txt_result:
        assert False, "No longer support kitti"


if __name__ == "__main__":
    main()

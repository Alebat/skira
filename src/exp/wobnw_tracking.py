from time import time

import cv2
import numpy as np

from src.exp.ski import SkiSequence
from tracking_wo_bnw.src.tracktor.frcnn_fpn import FRCNN_FPN
from tracking_wo_bnw.src.tracktor.reid.resnet import resnet50
from tracking_wo_bnw.src.tracktor.tracker import Tracker


def wobnw_tracking(video, detections, seed=12345,
                   obj_detect_model="faster_rcnn_fpn_training_mot_17/model_epoch_27.model",
                   reid_weights="reid/res50-mot17-batch_hard/ResNet_iter_25245.pth",
                   output_file=f'../data/tracks.txt',
                   tracker_cfg=None,
                   frame_split=None):
    if tracker_cfg is None:
        tracker_cfg = {
            # FRCNN score threshold for detections
            "detection_person_thresh": 0.2,
            # FRCNN score threshold for keeping the track alive
            "regression_person_thresh": 0.2,
            # NMS threshold for detection
            "detection_nms_thresh": 0.2,
            # NMS theshold while tracking
            "regression_nms_thresh": 0.3,
            # motion model settings
            "motion_model": {
                "enabled": False,
                # average velocity over last n_steps steps
                "n_steps": 1,
                # if true, only model the movement of the bounding box center.
                # If false, width and height are also modeled.
                "center_only": True,
            },
            # DPM or DPM_RAW or 0, raw includes the unfiltered (no nms) versions of the provided detections,
            # 0 tells the tracker to use private detections (Faster R-CNN)
            "public_detections": True,
            # How much last appearance features are to keep
            "max_features_num": 10,
            # Do camera motion compensation
            "do_align": True,
            # Which warp mode to use (cv2.MOTION_EUCLIDEAN, cv2.MOTION_AFFINE, ...)
            "warp_mode": str(cv2.MOTION_EUCLIDEAN),
            # maximal number of iterations (original 50)
            "number_of_iterations": 100,
            # Threshold increment between two iterations (original 0.001)
            "termination_eps": 0.00001,
            # Use siamese network to do reid
            "do_reid": True,
            # How much timesteps dead tracks are kept and considered for reid
            "inactive_patience": 20,
            # How similar do image and old track need to be to be considered the same person
            "reid_sim_threshold": 2.0,
            # How much IoU do track and image need to be considered for matching
            "reid_iou_threshold": 0.5,
        }
    if frame_split is None:
        frame_split = [0.0, 1.0]

    # set all seeds
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

    ##########################
    # Initialize the modules #
    ##########################

    # object detection
    print("Initializing object detector.")

    obj_detect = FRCNN_FPN(num_classes=2)
    obj_detect.load_state_dict(torch.load(obj_detect_model,
                               map_location=lambda storage, loc: storage))

    obj_detect.eval()
    obj_detect.cuda()

    # reid
    reid_network = resnet50(pretrained=False, output_dim=128)
    reid_network.load_state_dict(torch.load(reid_weights,
                                            map_location=lambda storage, loc: storage))
    reid_network.eval()
    reid_network.cuda()

    tracker = Tracker(obj_detect, reid_network, tracker_cfg)

    time_total = 0
    num_frames = 0
    seq = SkiSequence(video, detections)
    tracker.reset()
    start = time()

    print(f"Tracking: {seq}")
    data_loader = DataLoader(seq, batch_size=1, shuffle=False)
    for i, frame in enumerate(tqdm(data_loader)):
        if len(seq) * frame_split[0] <= i <= len(seq) * frame_split[1]:
            tracker.step(frame)
            num_frames += 1
    results = tracker.get_results()

    time_total += time() - start

    print(f"Tracks found: {len(results)}")
    print(f"Runtime for {seq}: {time() - start :.1f} s.")

    print(f"Writing predictions to: {output_file}")
    seq.write_results(results, output_file)
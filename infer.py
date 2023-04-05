import time
from ultralytics import YOLO
import numpy as np
import cv2
from tracker.byte_tracker import BYTETracker
import argparse

def make_parser():
    parser = argparse.ArgumentParser("ByteTrack Demo!")
    parser.add_argument("--track_thresh", type=float, default=0.3, help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")
    parser.add_argument('--aspect_ratio_thresh', type=float, default=1.6,
        help="threshold for filtering out boxes of which aspect ratio are above the given value.")
    parser.add_argument('--min_box_area', type=float, default=3000, help='filter out tiny boxes')
    parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")
    return parser

def get_color(idx):

    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)

    return color

def plot_tracking(image, tlwhs, obj_ids,  frame_id=0, fps=0., ids2=None):

    im = np.ascontiguousarray(np.copy(image))

    text_scale = 2
    text_thickness = 2
    line_thickness = 3

    cv2.putText(im, 'frame: %d fps: %.2f num: %d' % (frame_id, fps, len(tlwhs)),
                (0, int(15 * text_scale)), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), thickness=2)

    for i, tlwh in enumerate(tlwhs):
        x1, y1, w, h = tlwh
        intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
        obj_id = int(obj_ids[i])
        id_text = '{}'.format(int(obj_id))
        if ids2 is not None:
            id_text = id_text + ', {}'.format(int(ids2[i]))
        color = get_color(abs(obj_id))

        cv2.rectangle(im, intbox[0:2], intbox[2:4], color=color, thickness=line_thickness)
        cv2.putText(im, id_text, (intbox[0], intbox[1]), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255),
                    thickness=text_thickness)
    return im

def main(args,weights_path,video_path):
    model = YOLO(weights_path)
    tracker = BYTETracker(args, frame_rate=30)

    cap = cv2.VideoCapture(video_path)

    frame_id = 0
    results = []

    while True:
        ret_val, frame = cap.read()

        if ret_val:
            t0 = time.time()
            results_boxes = model(frame, iou=0.5, conf=0.3,imgsz=640)[0]
            boxes = results_boxes.boxes.cpu().numpy()
            print(len(boxes))
            if len(boxes) >0:
                bboxes = boxes.xyxy
                scores = boxes.conf

                online_targets = tracker.update(bboxes,scores)
                online_tlwhs = []
                online_ids = []
                online_scores = []
                for i, t in enumerate(online_targets):
                    # tlwh = t.tlwh
                    tlwh = t.tlwh_yolox
                    tid = t.track_id
                    vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
                    # if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                    if tlwh[2] * tlwh[3] > args.min_box_area:
                        online_tlwhs.append(tlwh)
                        online_ids.append(tid)
                        online_scores.append(t.score)
                        results.append(
                            f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                        )
                t1 = time.time()
                time_ = (t1 - t0) * 1000

                online_im = plot_tracking(frame, online_tlwhs, online_ids, frame_id=frame_id + 1,
                                          fps=1000. / time_)

                cv2.imshow("frame", online_im)

            else:
                t1 = time.time()
                time_ = (t1 - t0) * 1000
                cv2.putText(frame, 'frame: %d fps: %.2f num: %d' % (frame_id, 1000. / time_, 0),
                            (0, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), thickness=2)

                cv2.imshow("frame",frame)

            if cv2.waitKey(10) == 'q':
                break

            frame_id += 1
            t2 = time.time()
            print("infer and track time: {} ms".format((t2 - t0) * 1000))
            print()
        else:
            break


if __name__ =="__main__":
    args = make_parser().parse_args()

    weights_path = "./weights/yolov8n.pt"
    video_path = "/home/cai/project/person_bytetrack/videos/NOR_0000000_000000_20221208_101416_0003.mp4"

    main(args,weights_path,video_path)
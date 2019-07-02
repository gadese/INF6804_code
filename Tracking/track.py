from sort import *
from collections import defaultdict

import cv2


def write_tracks(file, id, frame, xmin, ymin, xmax, ymax):
    file.write('{} {} {} {} {} {}\n'.format(frame, id, xmin, ymin, xmax, ymax))


def read_gt(filename):
    if filename == -1:
        return defaultdict(list)
    gt = defaultdict(list)
    with open(filename, 'r') as file:
        for line in file:
            split = line.split(',')
            frame = int(split[0])
            id = int(split[1])
            x_min = int(float(split[2]))
            y_min = int(float(split[3]))
            x_max = x_min + int(float(split[4]))
            y_max = y_min + int(float(split[5]))
            gt[frame].append((x_min, y_min, x_max, y_max))
    return gt


def read_detections(filename):
    detections = defaultdict(list)
    with open(filename, 'r') as file:
        for line in file:
            split = line.split(' ')
            id = int(split[0])
            x_min = int(split[1])
            y_min = int(split[2])
            x_max = int(split[3])
            y_max = int(split[4])
            detections[id].append((x_min, y_min, x_max, y_max))
    return detections


def track(videopath, detections, hypo, size, gt):
    with open(hypo, 'w') as file:
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 0, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128),
                  (128, 0, 128), (128, 128, 0), (0, 128, 128)]

        video = cv2.VideoCapture(videopath)
        sort = Sort(max_age=5, min_hits=1)

        cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Video', size[0], size[1])

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        ret, current_frame = video.read()
        current_frame = cv2.resize(current_frame, size)
        vw = current_frame.shape[1]
        vh = current_frame.shape[0]
        print("Video size", vw, vh)
        outvideo = cv2.VideoWriter(videopath.replace(".mp4", "-det.mp4"), fourcc, 20.0, (vw, vh))

        frames = 0
        while True:
            ret, current_frame = video.read()
            if not ret:
                break
            frames += 1
            current_frame = cv2.resize(current_frame, size)
            frame_detections = detections[frames]
            frame_detections = np.array(frame_detections)
            gt_rect = gt[frames]

            tracked_objects = sort.update(frame_detections)

            for x1, y1, x2, y2, obj_id, cls_pred in tracked_objects:
                # color = colors[int(obj_id) % len(colors)]
                color = (0, 255, 0)
                cv2.rectangle(current_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 4)
                write_tracks(file, int(obj_id), int(frames), int(x1), int(y1), int(x2), int(y2))

            for box in gt_rect:
                cv2.rectangle(current_frame, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 4)

            cv2.imshow('Video', current_frame)
            outvideo.write(current_frame)
            # print(frames)
            # cv2.waitKey(0)
            ch = 0xFF & cv2.waitKey(1)
            if ch == 27:
                break

    cv2.destroyAllWindows()
    outvideo.release()
    return


def main():
    adl_video = 'videos/ADL-Rundle-6.mp4'
    adl_detections = 'detections/adl_rundle.txt'
    adl_tracks = 'track_results/adl_rundle.txt'
    adl_rundle_gt = 'gt/gt_adl_rundle.txt'
    adl_size = (1920, 1080)

    tud_video = 'videos/TUD-Stadtmitte.mp4'
    tud_detections = 'detections/tud_stadtmitte.txt'
    tud_tracks = 'track_results/tud_statdmitte.txt'
    tud_stadtmitte_gt = 'gt/gt_tud_statdmitte.txt'
    tud_size = (640, 480)

    venice_video = 'videos/Venice-2.mp4'
    venice_detections = 'detections/venice.txt'
    venice_tracks = 'track_results/venice.txt'
    venice_gt = 'gt/gt_venice.txt'
    venice_size = (1920, 1080)

    tp3_video = 'dataset/video.mp4'
    tp3_detections = 'results_det.txt'
    tp3_tracks = 'results.txt'
    tp3_gt = -1
    tp3_size = (1920, 1080)

    video = tud_video
    detections = read_detections(tud_detections)
    gt = read_gt(tud_stadtmitte_gt)
    hypo = tud_tracks
    size = tud_size

    track(video, detections, hypo, size, gt)


if __name__ == '__main__':
    main()


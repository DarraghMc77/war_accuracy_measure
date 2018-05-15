"""
    Simple Usage example (with 3 images)
"""
from mean_average_precision.detection_map import DetectionMAP
from mean_average_precision.utils.show_frame import show_frame
import numpy as np
import matplotlib.pyplot as plt
import json

classes = {'laptop': 0, 'cup': 1, 'tvmonitor': 2,  'mouse': 3, 'bottle': 4, 'diningtable': 5, 'keyboard': 6, 'person': 7, 'tv': 8,
           'cell phone': 9, 'car': 10, 'donut': 11, 'tie': 12, 'wine glass': 1, 'vase': 1, 'sink': 13, 'toilet': 14, 'chair': 15,
           'refrigerator': 16, 'remote': 17, 'toothbrush': 18, 'train': 19, 'cat': 20, 'microwave': 21, 'fire hydrant': 22, 'cake': 23, 'banana': 24, 'broccoli': 25, "pottedplant": 26, "bird": 27}

# classes_voc = {"aeroplane": 0,
#             "bicycle": 1,
#             "bird": 2,
#             "boat": 3,
#             "bottle": 4,
#             "bus": 5,
#             "car": 6,
#             "cat": 7,
#             "chair": 8,
#             "cow": 9,
#             "diningtable": 10,
#             "dog": 11,
#             "horse": 12,
#             "motorbike": 13,
#             "person": 14,
#             "pottedplant": 15,
#             "sheep": 16,
#             "sofa": 17,
#             "train": 18,
#             "tvmonitor": 19}

classes_voc = {"tvmonitor": 0,
            "person": 1,
            "bottle": 2,
            "diningtable": 3,
            "pottedplant": 4,
            "dog": 5,
            "car": 6,
            "bird": 7,
            "cat": 8}

def convert_bounding_boxes(bboxes):
    boxes = []
    for box in bboxes:
        if box:
            x1 = box[0] - box[2] / 2
            y1 = box[1] - box[3] / 2
            x2 = box[0] + box[2] / 2
            y2 = box[1] + box[3] / 2
            boxes.append([x1, y1, x2, y2])
        else:
            boxes.append(box)
    return boxes

pred_bb1 = np.array([])
pred_cls1 = np.array([])
pred_conf1 = np.array([])

gt_bb1 = np.array(convert_bounding_boxes([[341.80010986328125, 200.9195861816406, 493.32745361328125, 324.69921875],
                     [226.7739715576172, 376.61090087890625, 189.3741912841797, 289.5867919921875],
                     [574.0327758789062, 126.26085662841797, 212.88021850585938, 83.94528198242188]]))
gt_cls1 = np.array([0, 1, 2])


def read_json_file(file_path):
    od_frames = []
    with open(file_path) as f:
        for line in f:
            print(line.rstrip())
            if line != '[]\n':
                j_content = json.loads(line.rstrip())
                od_frames.append(j_content)

    return od_frames

def read_full_json_file(file_path):
    od_frames = []
    with open(file_path) as f:
        for line in f:
            print(line.rstrip())
            j_content = json.loads(line.rstrip())
            od_frames.append(j_content)

    return od_frames

def read_other_file(file_path):
    od_frames = []
    with open(file_path) as f:
        for line in f:
            print(line.rstrip())
            new_str = line.rstrip().replace("'", "\"")
            j_content = json.loads(new_str)
            od_frames.append(j_content)

    return od_frames

def convert_frame(json_frame):
    pre_bb1 = np.array(
        [[float(json_frame['x1']), float(json_frame['y1']), float(json_frame['x2']), float(json_frame['y2'])]])
    pre_cls1 = np.array([classes_voc[json_frame['id']]])
    pre_conf1 = np.array([float(json_frame['confidence'])])
    return (pre_bb1, pre_cls1, pre_conf1)


# def equate_results(pred_results, gt_results):
#     new_pred_results = []
#
#     index = 0
#     for i in range(630+1):
#         if pred_results[index]['image'] != i:
#             new_pred_results.append([])
#         else:
#             new_pred_results.append(pred_results[index])
#             index += 1
#
#     return new_pred_results

def convert_results(pred_results):
    image_indexes = []
    for res in pred_results:
        image_indexes.append(res[0]['imageNumber'])

    return image_indexes

def convert_results_(pred_results):
    image_indexes = []
    for res in pred_results:
        image_indexes.append(res[0]['imageNumber'])

    return image_indexes

def convert_gt_frame(json_frame):
    bb = []
    cls = []
    conf = []

    if(len(json_frame) == 0):
        bb.append([])
    else:
        for res in json_frame:
            bb.append([float(res['topleft']['x']), float(res['topleft']['y']),
                  float(res['bottomright']['x']), float(res['bottomright']['y'])])

            cls.append(classes_voc[res['label']])

            conf.append(float(res['confidence']))

    pre_bb2 = np.array(bb)
    pre_cls2 = np.array(cls)
    pre_conf1 = np.array(conf)
    return (pre_bb2, pre_cls2, pre_conf1)


def filter_gt_frames(gt, pred_indexes):

    filtered_gt = []


    for i in pred_indexes:
        found = False
        for j, dic in enumerate(gt):
            if dic:
                # add latency here
                if dic[0]['image'] == i:
                    filtered_gt.append(dic)
                    found = True
                    break
        if found == False:
            filtered_gt.append([])


    # for x in gt:
    #     if x[0]['image'] in pred_indexes:
    #         # with open("../test_files/filtered_gt.txt", "a") as myfile:
    #         #     myfile.write(str(x) + "\n")
    #
    #
    #         filtered_gt.append(x)
    return filtered_gt

def convert_all_frames(pred_results, gt, pred_indexes):
    frames = []

    gt = filter_gt_frames(gt, pred_indexes)

    for i in range(len(pred_results)):
        (pre_bb1, pre_cls1, pre_conf1) = convert_gt_frame(pred_results[i])
        (gt_bb1, gt_cls1, gt_conf1) = convert_gt_frame(gt[i])
        frames.append((pre_bb1, pre_cls1, pre_conf1, gt_bb1, gt_cls1))

    return frames


def no_gt_filter(pred_results, gt):
    frames = []

    for i in range(len(pred_results)):
        if i == 123:
            print("here")
        (pre_bb1, pre_cls1, pre_conf1) = convert_gt_frame(pred_results[i])
        (gt_bb1, gt_cls1, gt_conf1) = convert_gt_frame(gt[i])
        frames.append((pre_bb1, pre_cls1, pre_conf1, gt_bb1, gt_cls1))

    return frames

if __name__ == '__main__':

    # results = read_json_file("/Users/Darragh/College/Dissertation/mean_average_precision/test_files/mobile_results.txt")

    results = read_full_json_file("/Users/Darragh/College/Dissertation/mean_average_precision/test_files/trackPred.txt")
    # results = read_json_file("/Users/Darragh/College/Dissertation/mean_average_precision/test_files/mobile_results_newnew.txt")
    # other_results = read_other_file("/Users/Darragh/College/Dissertation/mean_average_precision/test_files/od-test_gt_new.txt")
    # results = read_json_file(
    #     "/Users/Darragh/College/Dissertation/mean_average_precision/test_files/pred.txt")
    other_results = read_other_file(
        "/Users/Darragh/College/Dissertation/mean_average_precision/test_files/gt.txt")
    # other_results = read_other_file(
    #     "../test_files/filtered_gt.txt")



    # frames = [(pre_bb1, pre_cls1, pre_conf1, pre_bb2, pre_cls2)]
    n_class = 3

    # frames = convert_all_frames(results, other_results, convert_results(results))

    frames = no_gt_filter(results, other_results)

    mAP = DetectionMAP(n_class)
    for i, frame in enumerate(frames):
        print("Evaluate frame {}".format(i))
        if(i == 213):
            print("here")

        if(i == 123):
            show_frame(*frame)
        mAP.evaluate(*frame)

    mAP.plot()
    plt.show()
    #plt.savefig("pr_curve_example.png")

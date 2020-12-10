from __future__ import division

import argparse
from typing import List

from torch.utils.data import DataLoader

from models import *
from utils.datasets import *
from utils.parse_config import *
from utils.utils import *


def evaluate(model, path, iou_thres, conf_thres, nms_thres, img_size, batch_size):
    model.eval()

    # Get dataloader
    dataset = ListDataset(path, img_size=img_size, augment=False, multiscale=False)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=1, collate_fn=dataset.collate_fn
    )

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred)
    for batch_i, (_, imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc="Detecting objects")):

        # Extract labels
        labels += targets[:, 1].tolist()
        # Rescale target
        targets[:, 2:] = xywh2xyxy(targets[:, 2:])
        targets[:, 2:] *= img_size

        imgs = Variable(imgs.type(Tensor), requires_grad=False)

        with torch.no_grad():
            outputs = model(imgs)
            outputs = non_max_suppression(outputs, conf_thres=conf_thres, nms_thres=nms_thres)

        sample_metrics += get_batch_statistics(outputs, targets, iou_threshold=iou_thres)

    # Concatenate sample statistics
    true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
    precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)

    return precision, recall, AP, f1, ap_class


def save_print_results(file_name: str, class_names: List[str], confidence_threshold: float, iou_threshold: float, nms_threshold: float,
                       precision: np.ndarray, recall: np.ndarray, ap: np.ndarray, ap_class, f1: float):
    print("Average Precisions:")
    for i, c in enumerate(ap_class):
        print(f"+ Class '{c}' ({class_names[c]}) - AP: {ap[i]}")
    print("Precisions:")
    for i, c in enumerate(ap_class):
        print(f"+ Class '{c}' ({class_names[c]}) - Precision: {precision[i]}")
    print("Recall:")
    for i, c in enumerate(ap_class):
        print(f"+ Class '{c}' ({class_names[c]}) - Recall: {recall[i]}")

    if not os.path.exists(file_name):
        with open(file_name, 'w') as f:
            f.write(
                "conf_thrs, iou_thres, nms_thres, mAP, AP0, AP1, AP2, AP3, AP4, "
                "precision0, precision1, precision2, precision3, precision4, "
                "recall0, recall1, recall2, recall3, recall4, f1\n")
    with open(file_name, 'a') as f:
        f.write(f"{confidence_threshold}, ")
        f.write(f"{iou_threshold}, ")
        f.write(f"{nms_threshold}, ")
        f.write(f"{ap.mean()}, ")
        for i, c in enumerate(ap_class):
            f.write(f"{ap[i]}, ")
        for i, c in enumerate(precision):
            f.write(f"{precision[i]}, ")
        for i, c in enumerate(recall):
            f.write(f"{recall[i]}, ")
        f.write(f"{f1}")
        f.write("\n")
    print(f"mAP: {ap.mean()}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=8, help="size of each image batch")
    parser.add_argument("--model_def", type=str, default="config/yolov3-carla.cfg", help="path to model definition file")
    parser.add_argument("--data_config", type=str, default="config/carla.data", help="path to data config file")
    parser.add_argument("--weights_path", type=str, default="weights/yolov3.weights", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/carla.names", help="path to class label file")
    parser.add_argument("--iou_thres", type=float, default=0.5, help="iou threshold required to qualify as detected")
    parser.add_argument("--conf_thres", type=float, default=0.1, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.5, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--out", type=str, default=None, help="output file")
    parser.add_argument("--multi_thres", action="store_true")
    opt = parser.parse_args()
    print(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_config = parse_data_config(opt.data_config)
    valid_path = data_config["valid"]
    class_names: List[str] = load_classes(data_config["names"])

    # Initiate model
    model = Darknet(opt.model_def).to(device)
    if opt.weights_path.endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(opt.weights_path)
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load(opt.weights_path))

    checkpoint_file_name: str = opt.weights_path.replace("weights/", "") \
        .replace("weights\\", "") \
        .replace(".cfg", "") \
        .replace("checkpoints/", "") \
        .replace("checkpoints\\", "") \
        .replace(".pth", "")
    eval_file_name = opt.out if opt.out else f"eval/eval_{checkpoint_file_name}.csv"

    if opt.multi_thres:
        n_steps_conf = int(math.ceil((1 / opt.conf_thres)))
        n_steps_iou = int(math.ceil((1 / opt.iou_thres)))
        n_steps_nms = int(math.ceil((1 / opt.nms_thres)))
        for i in range(n_steps_conf + 1):
            confidence_thrs: float = (i * opt.conf_thres) + (0.001 if i == 0 else 0) - (0.001 if i == n_steps_conf else 0)
            for j in range(n_steps_iou + 1):
                iou_threshold: float = (j * opt.iou_thres) + (0.001 if j == 0 else 0) - (0.001 if j == n_steps_iou else 0)
            # for k in range(n_steps_nms + 1):
            #     nms_threshold: float = (k * opt.nms_thres) + (0.001 if k == 0 else 0) - (0.001 if k == n_steps_nms else 0)
                nms_threshold = opt.nms_thres
                # iou_threshold = opt.iou_thres
                print(f"Compute metrics for confidence threshold {confidence_thrs}, iou threshold {iou_threshold}, mns threshold {nms_threshold}")
                precision, recall, AP, f1, ap_class = evaluate(
                    model,
                    path=valid_path,
                    iou_thres=iou_threshold,
                    conf_thres=confidence_thrs,
                    nms_thres=nms_threshold,
                    img_size=opt.img_size,
                    batch_size=8,
                )
                save_print_results(file_name=eval_file_name, class_names=class_names, confidence_threshold=confidence_thrs,
                                   iou_threshold=iou_threshold, nms_threshold=nms_threshold, precision=precision,
                                   recall=recall, ap=AP, ap_class=ap_class, f1=f1)
    else:
        precision, recall, AP, f1, ap_class = evaluate(
            model,
            path=valid_path,
            iou_thres=opt.iou_thres,
            conf_thres=opt.conf_thres,
            nms_thres=opt.nms_thres,
            img_size=opt.img_size,
            batch_size=8,
        )
        save_print_results(file_name=eval_file_name, class_names=class_names, confidence_threshold=opt.conf_thres,
                           iou_threshold=opt.iou_thres, nms_threshold=opt.nms_thres, precision=precision,
                           recall=recall, ap=AP, ap_class=ap_class, f1=f1)


if __name__ == "__main__":
    main()

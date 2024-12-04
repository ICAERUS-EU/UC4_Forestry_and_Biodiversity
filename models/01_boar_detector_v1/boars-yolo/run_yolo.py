import os
import sys

# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, os.path.join(os.path.dirname(__file__)))

from pathlib import Path
from utils.torch_utils import select_device
from utils.general import check_img_size, non_max_suppression
from models.common import DetectMultiBackend
import cv2
import numpy as np
from utils.augmentations import letterbox
import torch
from utils.plots import Annotator, colors, save_one_box


class YoloModel:
    """Initialize and load YOLOv5 model for inference. A wrapper for running the model. 
    """

    def __init__(self, data_reader, weights, device='cpu', dnn=False, imgsz=(640, 640),
                 half=False, conf_thres=0.68, iou_thres=0.45, max_det=1000, classes=None, out_dir='results',
                 line_thickness=2):
        """
        Args:
            data_reader (AbstractReader): A reader is given as an input
            weights (str): path to yolo model weights
            device (str, optional): Set the divice for torch to use. Defaults to 'cpu'.
            dnn (bool, optional): Use dnn for opencv functions. Defaults to False.
            imgsz (tuple, or int, optional): Input image size reference. Defaults to (640, 640).
            half (bool, optional): Set to use half precission float calculation. Defaults to False.
            conf_thres (float, optional): Confidence threshold. Defaults to 0.25.
            iou_thres (float, optional): NMS IOU threshold. Defaults to 0.45.
            max_det (int, optional): Maximum num of detections in image. Defaults to 1000.
            classes (_type_, optional): Class filter. Defaults to None.
            out_dir (str, optional): output folder (created inside input folder). Defaults to 'results'.
            line_thickness (int, optional): _description_. Defaults to 2.
        """
        self.dr = data_reader
        self.weights = weights
        self.device = device
        self.dnn = dnn
        self.imgsz = imgsz
        if isinstance(self.imgsz,int):
            self.imgsz = (self.imgsz, self.imgsz)
        self.half = half
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_det = max_det
        self.classes = classes
        self.out_dir = out_dir
        self.line_thickness = line_thickness

    @staticmethod
    def _pred_parser(pred, allowed_classes=[1]):
        """Convert predictions to supported format. Filter by class and drop class data.

        Args:
            pred (list): list of predictions from yolo model
        """
        if len(pred) == 1:
            dt = pred[0].numpy()
            if dt.shape[0] == 0:  # check if prediction is empty
                return []
            assert dt.shape[-1] == 6, "prediction shape assertion failed. Possibly wrong prediction file provided"
            check = []
            for val in dt:  # filter classes
                if int(val[-1]) in allowed_classes:
                    check.append(True)
                else:
                    check.append(False)
            return dt[check, :-1]
        else:
            raise NotImplementedError

    def _load_model(self):
        """Load yolo model using given parameters. Run once only.
        """
        # set device, load model, get info about model. Taken from yolo code
        self.device = select_device(self.device)
        model = DetectMultiBackend(
            self.weights, device=self.device, dnn=self.dnn)
        stride, pt, jit, engine, self.names = model.stride, model.pt, model.jit, model.engine, model.names
        self.imgsz = check_img_size(self.imgsz, s=stride)  # check image size
        self.model = model
        self.stride = stride

        # set precision if required. Taken from yolo code
        # half precision only supported by PyTorch on CUDA
        self.half &= (pt or jit or engine) and self.device.type != 'cpu'
        if pt or jit:
            self.model.model.half() if self.half else self.model.model.float()

    def _inferencer_preprocessing(self, data):
        """Preprocess image for inference, taken from yolo detect.py file.

        Args:
            data (np.array): Original image data from reader

        Returns:
            np.array: augmented image
        """

        # Padded resize
        img = letterbox(data, self.imgsz, stride=self.stride, auto=True)[0]

        # Convert
        try:
            img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        except ValueError:
            if len(img.shape) == 2:
                img = np.expand_dims(img, axis=2)
                img = np.repeat(img, 3, axis=2)
                img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        return img

    def _generate_out_path(self, path):  # Nenaudojama
        """Generate image output path.

        Args:
            path (str): Path to current image

        Returns:
            str: Path to write created image to
        """
        p = Path(path)
        parent_path = str(p.parent)
        filename = p.name
        out_dir = os.path.join(parent_path, self.out_dir)
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        return os.path.join(out_dir, filename)

    def _inference(self, im):
        """Run yolo model on image

        Args:
            im (np.array): augmented image

        Returns:
            list: list of tensors of shape (n, 6), where n is the number predicted objects
        """
        # taken from yolo detect.py
        im = torch.from_numpy(im).to(self.device)
        im = im.half() if self.half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

        # Inference
        pred = self.model(im, augment=False, visualize=False)

        # NMS
        pred = non_max_suppression(
            pred, self.conf_thres, self.iou_thres, self.classes, False, max_det=self.max_det)
        return pred

    def _image_out(self, pred, img0, path):  # Nenaudojama
        """Generate anotated image from predictions

        Args:
            pred (list): Predictions from yolo model
            img0 (np.array): original image
            path (str): path to image output location
        """
        # taken from yolo detect.py
        # Process predictions
        for det in pred:  # per image
            annotator = Annotator(
                img0, line_width=self.line_thickness, example="")
            if len(det):
                # det[:, :4] = scale_coords(im.shape[2:], det[:, :4], img0.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class
                    label = f'{self.names[c]} {conf:.2f}'  # bbox labeling
                    label = ""
                    annotator.box_label(xyxy, label, color=colors(c, True))
            img0 = annotator.result()
            cv2.imwrite(path, img0)
        return True

    def image_out_bbox(self, bbox, img0, path):
        """Generate annotated image from predictions

        Args:
            bbox (list): Bbox from yolo model
            img0 (np.array): original image
            path (str): path to image output location
        """
        # taken from yolo detect.py
        # Process predictions
        annotator = Annotator(
            img0, line_width=self.line_thickness, example="")
        if len(bbox) > 0:
            annotator.box_label(bbox, "", color=colors(1, True))
        img0 = annotator.result()
        cv2.imwrite(path, img0)
        return True

    def run(self):
        """Run yolo model on the data from reader

        Returns:
            list: bboxes from predictions [x1 y1 x2 y2 percentage] (boars only)
        """
        # yolo dataset grazina: path, np.array img processed, np.array img0 origninal, video capture, log string 
        bboxes = []
        for input_data in self.dr:
            img = self._inferencer_preprocessing(input_data)
            pred = self._inference(img)
            pred = YoloModel._pred_parser(pred)
            bboxes.append(pred)
        return bboxes

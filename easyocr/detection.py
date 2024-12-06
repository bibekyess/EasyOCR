import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from PIL import Image
from collections import OrderedDict

import cv2
import numpy as np
from .craft_utils import getDetBoxes, adjustResultCoordinates
from .imgproc import resize_aspect_ratio, normalizeMeanVariance
from .craft import CRAFT
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

def test_net(canvas_size, mag_ratio, net, image, text_threshold, link_threshold, low_text, poly, device, estimate_num_chars=False):
    if isinstance(image, np.ndarray) and len(image.shape) == 4:  # image is batch of np arrays
        image_arrs = image
    else:                                                        # image is single numpy array
        image_arrs = [image]

    # Assuming image_arrs is a list of NumPy arrays (original images)
    img_processed_list = []

    # Preprocessing each image without resizing
    for img in image_arrs:
        # Apply normalization directly without resizing
        normalized_img = normalizeMeanVariance(img)
        # Rearrange dimensions to match PyTorch input format (C, H, W)
        img_processed = np.transpose(normalized_img, (2, 0, 1))
        img_processed_list.append(img_processed)

    # Convert to PyTorch tensor
    x = torch.from_numpy(np.array(img_processed_list))
    
    ratio_h = ratio_w = 1
    
    if 'openvino' in device:
        x = x.to('cpu')
        import time
        start_time = time.time()
        # forward pass
        # res=net.infer_new_request({0: x})
        res = net([x])
        logging.info(f'Detection timing: {time.time()-start_time}')
        y=torch.tensor(res[0])
    else:
        x = x.to(device)
        # forward pass
        with torch.no_grad():
            y, feature = net(x)

    boxes_list, polys_list = [], []
    for out in y:
        # make score and link map
        score_text = out[:, :, 0].cpu().data.numpy()
        score_link = out[:, :, 1].cpu().data.numpy()

        # Post-processing
        boxes, polys, mapper = getDetBoxes(
            score_text, score_link, text_threshold, link_threshold, low_text, poly, estimate_num_chars)

        # coordinate adjustment
        boxes = adjustResultCoordinates(boxes, ratio_w, ratio_h)
        polys = adjustResultCoordinates(polys, ratio_w, ratio_h)
        if estimate_num_chars:
            boxes = list(boxes)
            polys = list(polys)
        for k in range(len(polys)):
            if estimate_num_chars:
                boxes[k] = (boxes[k], mapper[k])
            if polys[k] is None:
                polys[k] = boxes[k]
        boxes_list.append(boxes)
        polys_list.append(polys)

    return boxes_list, polys_list

def get_detector(trained_model, device='cpu', quantize=True, cudnn_benchmark=False):
    net = CRAFT()

    if device == 'cpu':
        net.load_state_dict(copyStateDict(torch.load(trained_model, map_location=device)))
        if quantize:
            try:
                torch.quantization.quantize_dynamic(net, dtype=torch.qint8, inplace=True)
            except:
                pass
        net.eval()

    elif 'openvino' in device:
        import openvino as ov
        import os
        net.load_state_dict(copyStateDict(torch.load(trained_model, map_location='cpu')))
        net.eval()
        core = ov.Core()

        cache_dir = os.getenv('EASYOCR_MODULE_PATH', os.path.expanduser('~/.EasyOCR/cache'))
        ov_model_path = os.path.join(cache_dir, "easy_ocr_detection", "ov_model.xml")
        ov_model_bin_path = os.path.join(cache_dir, "easy_ocr_detection", "ov_model.bin")

        if os.path.exists(ov_model_path) and os.path.exists(ov_model_bin_path):
            logging.info("Loading OpenVINO model from file ...")
            net_ov = core.read_model(model=ov_model_path)
        else:
            logging.info("Converting Torch model to OpenVINO")
            dummy_inp = torch.rand(1, 3, 608, 800)
            net_ov = ov.convert_model(net, example_input=dummy_inp)
            logging.info("Saving converted OpenVINO model to file ...")
            ov.save_model(net_ov, ov_model_path)            
        logging.info("Compiling OpenVINO model ...")
        static_shape = [1, 3, 1130, 800]  # N, C, H, W 
        net_ov.reshape({0: static_shape})  # Input 0 will have this shape
        net=core.compile_model(net_ov, device_name='AUTO:NPU,GPU,CPU')

    else:
        net.load_state_dict(copyStateDict(torch.load(trained_model, map_location=device)))
        net = torch.nn.DataParallel(net).to(device)
        cudnn.benchmark = cudnn_benchmark
        net.eval()
    return net

def get_textbox(detector, image, canvas_size, mag_ratio, text_threshold, link_threshold, low_text, poly, device, optimal_num_chars=None, **kwargs):
    result = []
    estimate_num_chars = optimal_num_chars is not None
    bboxes_list, polys_list = test_net(canvas_size, mag_ratio, detector,
                                       image, text_threshold,
                                       link_threshold, low_text, poly,
                                       device, estimate_num_chars)
    if estimate_num_chars:
        polys_list = [[p for p, _ in sorted(polys, key=lambda x: abs(optimal_num_chars - x[1]))]
                      for polys in polys_list]

    for polys in polys_list:
        single_img_result = []
        for i, box in enumerate(polys):
            poly = np.array(box).astype(np.int32).reshape((-1))
            single_img_result.append(poly)
        result.append(single_img_result)

    return result

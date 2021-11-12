import collections
import torch
import os
import easyocr
import logging


def test_net(canvas_size, mag_ratio, net, image, text_threshold, link_threshold, low_text, poly, device, estimate_num_chars=False):
    if isinstance(image, np.ndarray) and len(image.shape) == 4:  # image is batch of np arrays
        image_arrs = image
    else:                                                        # image is single numpy array
        image_arrs = [image]

    img_resized_list = []
    # resize
    target_ratio = 0.
    for img in image_arrs:
        img_resized, target_ratio, _ = easyocr.imgproc.resize_aspect_ratio(img, canvas_size,
                                                                      interpolation=cv2.INTER_LINEAR,
                                                                      mag_ratio=mag_ratio)
        img_resized_list.append(img_resized)
    ratio_h = ratio_w = 1 / target_ratio
    # preprocessing
    x = [np.transpose(normalizeMeanVariance(n_img), (2, 0, 1))
         for n_img in img_resized_list]
    x = torch.from_numpy(np.array(x))
    x = x.to(device)

    # forward pass
    with torch.no_grad():
        # feature 확인해보기
        # U-net 그림 참조
        # https://towardsdatascience.com/neural-networks-intuitions-6-east-5892f85a097
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


def example_copyStateDict(state_dict):
    # print(list(state_dict.keys()))  # -> ['module.basenet.slice1.0.weight', 'module.basenet.slice1.0.bias', ...]
    # startswith('시작하는 문자', '시작 지점') : 특정 문자열 찾기
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = collections.OrderedDict()
    for k, v in state_dict.items():
        # key name : 'module.basenet.slice1.0.weight' -> 'basenet.slice1.0.weight'
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict


def example_torch_load():
    # logger = logging.getLogger(name=__name__)
    # logger.setLevel(logging.INFO)

    trained_model = os.path.expanduser('~/.EasyOCR/model/craft_mlt_25k.pth')
    net = easyocr.craft.CRAFT()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Pytorch Device was set : {device.upper()}')

    loaded: collections.OrderedDict = torch.load(f=trained_model, map_location=device)
    # print(type(loaded))  # -> <class 'collections.OrderedDict'>
    # print(loaded)  # -> 사전 학습된 각 layer의 w, b값이 출력
    new_loaded = example_copyStateDict(state_dict=loaded)
    net.load_state_dict(state_dict=new_loaded)  # load_state_dict method부터 파악 시작하기
    print(net)  #  신경망 구조 확인

    # net : detector
    net = torch.nn.DataParallel(net).to(device)
    cudnn.benchmark = cudnn_benchmark
    net.eval()

    # process getTextBox
    min_size = 20
    text_threshold = 0.7
    low_text = 0.4
    link_threshold = 0.4
    canvas_size = 2560
    mag_ratio = 1.
    slope_ths = 0.1
    ycenter_ths = 0.5
    height_ths = 0.5
    width_ths = 0.5
    add_margin = 0.1
    reformat = True
    optimal_num_chars = None
    estimate_num_chars = optimal_num_chars is not None

    bboxes_list, polys_list = test_net(canvas_size=canvas_size, mag_ratio=mag_ratio, net=net, image=image,
                                       text_threshold=text_threshold, link_threshold=link_threshold,
                                       low_text=low_text, poly=False, device=device, estimate_num_chars=estimate_num_chars)


if __name__ == '__main__':
    example_torch_load()

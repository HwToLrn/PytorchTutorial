import torch
import easyocr
import numpy as np
import cv2
import torch.nn.functional as F


def test_net(canvas_size, mag_ratio, net, image, text_threshold, link_threshold, low_text, poly, device, estimate_num_chars=False):
    pass


# easyocr.craft.CRAFT() code comprehension
def main():
    # data
    image = '../ImageDatas/image1.jpg'
    image, _ = easyocr.utils.reformat_input(image=image)
    print('image.shape : ', image.shape)
    print('type(image) : ', type(image))

    # hyper parameters
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
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # network
    base_net = easyocr.model.modules.vgg16_bn(pretrained=False, freeze=False)
    base_net.to(device)
    # print(base_net)

    # make image's dimension same
    if isinstance(image, np.ndarray) and len(image.shape) == 4:  # image is batch of np arrays
        image_arrs = image
    else:                                                        # image is single numpy array
        image_arrs = [image]

    # resize
    img_resized_list = []
    target_ratio = 0.
    for img in image_arrs:
        img_resized, target_ratio, _ = easyocr.imgproc.resize_aspect_ratio(img, canvas_size,
                                                                           interpolation=cv2.INTER_LINEAR,
                                                                           mag_ratio=mag_ratio)
        img_resized_list.append(img_resized)
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = [np.transpose(easyocr.imgproc.normalizeMeanVariance(n_img), (2, 0, 1))
         for n_img in img_resized_list]
    x = torch.from_numpy(np.array(x))
    x = x.to(device)

    source = base_net(x)
    # for idx in range(5):
    #     print(source[idx])
    # print()

    # torch.cat() : https://sanghyu.tistory.com/85
    print(source[0].shape, '\tsource[0].shape')  # torch.Size([1, 1024, 14, 16])
    print(source[1].shape, '\tsource[1].shape')  # torch.Size([1, 512, 14, 16])
    fst_unet = torch.cat(tensors=[source[0], source[1]], dim=1)
    print(fst_unet.shape, '\t0, 1 concatenated shape')  # torch.Size([1, 1536, 14, 16])
    print('='*50)
    print(source[2].size(), '\tsource[2].size()')  # -> torch.Size([1, 512, 28, 32])
    # fst_unet과 source[2]의 shape을 맞추기 위해 F.interpolate()를 사용한다.
    scd_unet = F.interpolate(input=fst_unet, size=sources[2].size()[2:], mode='bilinear', align_corners=False)


if __name__ == '__main__':
    main()
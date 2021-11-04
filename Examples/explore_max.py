

if __name__ == '__main__':
    canvas_size = 2560
    target_size = 2560
    height = 720
    width = 680
    print('max(height, width) :', max(height, width))
    ratio = target_size / max(height, width)
    print('ratio :', ratio)
    # target_h32 = target_h + (32 - target_h % 32)
    target_h, target_w = int(height * ratio), int(width * ratio)
    print('target_h, target_w :', target_h, target_w)
    target_w32 = target_w + (32 - target_w % 32)
    print('target_w % 32 :', target_w % 32)
    print('target_w32 :', target_w32)


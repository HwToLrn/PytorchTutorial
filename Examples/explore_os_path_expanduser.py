import os


if __name__ == '__main__':
    # Home directory -> C:\Users\pksmb
    print(os.path.expanduser('~'))

    print(os.path.expanduser("~/.EasyOCR/"))  # return C:\Users\pksmb/.EasyOCR/
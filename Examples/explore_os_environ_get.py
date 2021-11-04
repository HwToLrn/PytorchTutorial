import os


if __name__ == '__main__':
    # 운영체제에 설정해놓은 환경변수를 가져올 수 있는 method
    # 설정된 key가 아니면 None을 반환한다.
    print(os.environ.get('PATH'))
    print(os.environ.get('EASYOCR_MODULE_PATH'))
    print(os.environ.get('MODULE_PATH'))
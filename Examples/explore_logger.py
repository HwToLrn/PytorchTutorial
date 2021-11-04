import logging

mylogger = logging.getLogger(__name__)


# 참조 Link : https://hamait.tistory.com/880
if __name__ == '__main__':
    mylogger.warning('you writed the message that send your user the warning')
    stream_hander = logging.StreamHandler()
    mylogger.addHandler(stream_hander)

    # log level을 낮춰주지 않아서 출력이 안됨
    # 자세한 사항은 link에서 확인
    mylogger.info('fisrt server start!!!')

    mylogger.setLevel(logging.INFO)
    mylogger.info("second server start!!!")
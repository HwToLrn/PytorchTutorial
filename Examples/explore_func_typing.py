from typing import Union


# 아래 방식보다는 def add(x: Union[int, float], y: Union[int, str]): 형태가 더 유용해 보임
# (내가 발견한) 차이점 : PyCharm 환경에서 Problems warning을 띄워주지 않음
def add(x: {int, float}, y: {int, str}):
    if isinstance(y, str):
        y = float(y)
    else:
        raise ValueError('y argument must be either "int" or "str" type')
    c = x + y
    return c


def main():
    result = add(x=2., y='5.')
    print(f'result : {result}')


if __name__ == '__main__':
    main()
import collections


# collections Link : https://docs.python.org/ko/3.9/library/collections.html
def example_namedtuple():
    # collections.namedtuple() : Returns a new subclass of tuple with named fields
    # 이름 붙은 필드를 갖는 튜플 서브 클래스를 만들기 위한 팩토리 함수
    Point = collections.namedtuple(typename='MyPoint', field_names=['x', 'y'])
    print(Point.__doc__)  # -> MyPoint(x, y)
    p1 = Point(x=11, y=22)
    print(p1)  # -> MyPoint(x=11, y=22)
    print(p1[0], p1[1])  # -> 11 22
    print(p1.x, p1.y)  # -> 11 22
    p2 = Point(x=1, y=2)
    print(p2)  # -> MyPoint(x=1, y=2)


def example_orderedDict():
    # collections.OrderedDict() : Dictionary that remembers insertion order
    # python 3.6 이전에서는 사전에 데이터를 삽입된 순서대로 데이터를 획득할 수가 없었기에
    # 무작위 순서로 데이터를 얻게 되는 일이 빈번했었다고 한다.
    # 순서를 필요로하는 경우를 위해 삽입된 순서를 기억하는 Ordered dictionary(정렬된 dict)가 나왔다.
    # 그러나, python 3.6부터는 print로 보면 dict()이나 OrderedDict()이나 동일하게 순서가 적용되어 동작한다.

    # 하위 호환성 보장 측면에서 가급적 데이터의 순서가 중요한 경우에는 OrderedDict()를 사용하는 것이 권장된다.
    # 차이점은 동등성 비교에서 나타난다. 아래 '동등성 비교' 예시를 참조하자.
    ord_dict = collections.OrderedDict()
    comp_dict = dict()

    # 순서대로 하나하나 넣는다.
    ord_dict['A'], comp_dict['A'] = 0, 0
    ord_dict['B'], comp_dict['B'] = 1, 1
    ord_dict['C'], comp_dict['C'] = 2, 2
    ord_dict['D'], comp_dict['D'] = 3, 3

    for key, value in comp_dict.items():
        print(f'{{{key}:{value}}}', end=' ')  # -> {A:0} {B:1} {C:2} {D:3}
    print()
    for key, value in ord_dict.items():
        print(f'{{{key}:{value}}}', end=' ')  # -> {A:0} {B:1} {C:2} {D:3}
    print()

    # 동등성 비교
    dict_data1 = {'A': 0, 'B': 1, 'C': 2}  # == dict(A=0, B=1, C=2)
    dict_data2 = {'A': 0, 'C': 2, 'B': 1}
    print(dict_data1 == dict_data2)  # -> True

    # 순서까지 고려하기 때문에 False 출력
    dict_data1 = collections.OrderedDict(dict(A=0, B=1, C=2))
    dict_data2 = collections.OrderedDict(dict(A=0, C=2, B=1))
    print(dict_data1 == dict_data2)  # -> False


if __name__ == '__main__':
    # example_namedtuple()
    example_orderedDict()
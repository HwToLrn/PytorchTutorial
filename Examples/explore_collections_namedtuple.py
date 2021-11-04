import collections


def main():
    # collections.namedtuple() : Returns a new subclass of tuple with named fields
    Point = collections.namedtuple(typename='MyPoint', field_names=['x', 'y'])
    print(Point.__doc__)
    p1 = Point(x=11, y=22)
    print(p1)
    print(p1[0], p1[1])
    print(p1.x, p1.y)
    p2 = Point(x=1, y=2)
    print(p2)


if __name__ == '__main__':
    main()
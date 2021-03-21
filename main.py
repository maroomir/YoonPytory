import yoonpytory


def process_test_dir():
    # Get Direction list
    list_dir = yoonpytory.dir2D.get_clock_directions()
    print("Number of clock direction : {0}".format(len(list_dir)))
    print("Directions : {0}".format(list_dir))
    # Go direction per "Ordering"
    direction = list_dir[0]
    vector = yoonpytory.vector2D.direction_vector(direction)
    for i in range(10):
        print("DIR {0} = ".format(i) + direction.__str__())
        print("VECTOR {0} = ".format(i) + vector.__str__())
        direction += "order"
        vector += "order"
    # Add direction directly
    print("LAST DIR = " + direction.__str__())
    direction += yoonpytory.dir2D.RIGHT
    print("ADD DIR = " + direction.__str__())


def process_test_vector():
    vector1 = yoonpytory.vector2D(10, 15)
    vector2 = yoonpytory.vector2D(13, 19)
    print("VEC 1 = {0}, VEC 2 = {1}, DISTANCE = {2}".format(vector1.to_tuple(), vector2.to_tuple(),
                                                            vector1.distance(vector2)))
    scale_vector = vector1.scale(2, 2)
    move_vector = yoonpytory.vector2D(10, 10, step=5)
    print("Scale VEC 1 = " + scale_vector.__str__())
    # Move vector to direction
    for i in range(10):
        print("MOVE {0} = ".format(i) + move_vector.__str__())
        move_vector += yoonpytory.dir2D.TOP_RIGHT


def process_test_rect():
    vector = yoonpytory.vector2D(10, 10, step=5)
    list_vector = [vector]
    for i_dir in yoonpytory.dir2D.get_square_directions():
        print("INSERT VECTOR = " + vector.__str__())
        vector += i_dir
        print("DIR TO = " + i_dir.__str__())
        list_vector.append(vector)
    rect1 = yoonpytory.rect2D(list=list_vector)
    rect2 = yoonpytory.rect2D(x=0, y=0, width=15, height=15)
    print("RECT FROM TUPLES = " + rect1.__str__())
    print("OBJECT RECT = " + rect2.__str__())
    print("SUM = " + (rect1 + rect2).__str__())


def process_test_line():
    # Move vector to direction
    move_vector = yoonpytory.vector2D(5, 5, step=5)
    list_vector = []
    for i in range(10):
        print("MOVE {0} = ".format(i) + move_vector.__str__())
        list_vector.append(move_vector.__copy__())
        move_vector += yoonpytory.dir2D.TOP_RIGHT
    line1 = yoonpytory.line2D(list_vector[0], list_vector[1], list_vector[2], list_vector[3], list_vector[4])
    line2 = yoonpytory.line2D(list=list_vector)
    other_vector = yoonpytory.vector2D(1, -1)
    print(line1.__str__())
    print(line2.__str__())
    print("DIST = {}".format(line1.distance(other_vector)))
    print((line1 + line2).__str__())


if __name__ == '__main__':
    process_test_dir()
    process_test_vector()
    process_test_rect()
    process_test_line()

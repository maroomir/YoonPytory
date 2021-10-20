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
    move_vector = yoonpytory.vector2D(10, 10)
    print("Scale VEC 1 = " + scale_vector.__str__())
    # Move vector to direction
    for i in range(10):
        print("MOVE {0} = ".format(i) + move_vector.__str__())
        move_vector += yoonpytory.dir2D.TOP_RIGHT


def process_test_rect():
    vector = yoonpytory.vector2D(10, 10)
    list_vector = [vector]
    for i_dir in yoonpytory.dir2D.get_square_directions():
        print("INSERT VECTOR = " + vector.__str__())
        for i in range(5):
            vector += i_dir
        print("DIR TO = " + i_dir.__str__())
        list_vector.append(vector)
    rect1 = yoonpytory.rect2D.from_list(args=list_vector)
    rect2 = yoonpytory.rect2D(x=0, y=0, width=15, height=15)
    print("RECT FROM TUPLES = " + rect1.__str__())
    print("OBJECT RECT = " + rect2.__str__())
    print("SUM = " + (rect1 + rect2).__str__())


def process_test_line():
    # Move vector to direction
    move_vector = yoonpytory.vector2D(5, 5)
    list_vector = []
    for i in range(10):
        print("MOVE {0} = ".format(i) + move_vector.__str__())
        list_vector.append(move_vector.__copy__())
        move_vector += yoonpytory.dir2D.TOP_RIGHT
    line1 = yoonpytory.line2D.from_vectors(list_vector[0], list_vector[1], list_vector[2], list_vector[3], list_vector[4])
    line2 = yoonpytory.line2D.from_list(list_vector)
    other_vector = yoonpytory.vector2D(1, -1)
    print(line1.__str__())
    print(line2.__str__())
    print("DIST = {}".format(line1.distance(other_vector)))
    print((line1 + line2).__str__())


def process_single_layer_perception():
    net = yoonpytory.neuron()
    net.load_source(file_path='./data/slp/twoGaussians.npz')
    net.load_weight(file_path='./data/slp/weight.npz')
    net.train(epoch=2000, is_init_weight=True, is_run_test=False)
    net.process()
    net.save_weight(file_path='./data/slp/weight.npz')


def process_multi_layer_perception():
    net = yoonpytory.network()
    net.load_source(file_path='./data/mlp/spirals.npz')
    net.load_weight(file_path='./data/mlp/weights.npz')
    # net.train(nCountEpoch=1000, nSizeLayer=100, nOrder=10, bInitWeight=False, bRunTest=False)
    net.process()
    net.show_plot()
    net.save_weight(file_path='./data/mlp/weights.npz')


if __name__ == '__main__':
    print("Select the sample process")
    print("1. Direction")
    print("2. Vector")
    print("3. Rect")
    print("4. Line")
    print("5. SLP")
    print("6. MLP")
    process = input(">>")
    process = process.lower()
    if process == "1" or "direction":
        process_test_dir()
    elif process == "2" or "vector":
        process_test_vector()
    elif process == "3" or "rect":
        process_test_rect()
    elif process == "4" or "line":
        process_test_line()
    elif process == "5" or "slp":
        process_single_layer_perception()
    elif process == "6" or "mlp":
        process_multi_layer_perception()

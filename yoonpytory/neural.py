import matplotlib.pyplot
import numpy


def sigmoid(array: numpy.ndarray):
    return numpy.exp(-numpy.logaddexp(0, -array))


def logistic_regression(array_x: numpy.ndarray, array_w: numpy.ndarray):  # Output : Calculate output (pY)
    # X is input (row-wise sample)
    # W is weight
    array_v = numpy.matmul(array_w, array_x.transpose())
    array_v = array_v.transpose()
    array_y = sigmoid(array_v)
    return array_y


def gradient_descent(array_x: numpy.ndarray, array_w: numpy.ndarray, array_y: numpy.ndarray,
                     alpha=0.001):  # Output : improved weight (W+1)
    # X is input
    count_x, dim_x = array_x.shape
    # W is weight estimated (not constructed)
    count_w, dim_w = array_w.shape
    # Y is target output (Right answer)
    count_y, dim_y = array_y.shape
    if count_w != 1 and dim_y != 1:
        raise Exception("The layer count is not only one")
    if count_x != count_y:
        raise Exception("Array X count {0}, Delta count {1} is not equal".format(count_x, count_y))
    if dim_x != dim_w:
        raise Exception("X Dimension {0}, W Dimension {1} is not equal".format(dim_x, dim_w))
    for i in range(count_x):
        x = array_x[i, :]  # iInput length = nDimension
        target = array_y[i]  # Size = 1
        # feedforward process
        y = logistic_regression(x, array_w)
        # backward process
        error = target - y
        gradient = alpha * error * x
        array_w = array_w + gradient.transpose()
    return array_w


def back_propagation(array_x: numpy.ndarray, weights: list, array_y: numpy.ndarray,
                     alpha=0.01):  # Output : improved weight List(W+1)
    # X is input
    # W is weight estimated (not constructed)
    # Y is target output (Right answer)
    count_x, dim_x = array_x.shape
    for i in range(count_x):
        x = numpy.array(array_x[i, :], ndmin=2)
        target = numpy.array(array_y[i], ndmin=2)
        # feedforward process
        y = []
        layer = x.copy()
        for j in range(0, len(weights), 1):
            assert isinstance(weights[j], numpy.ndarray)
            layer = logistic_regression(layer.copy(), weights[j])
            y.append(layer.copy())
        # backward process
        errors = [numpy.ndarray] * len(weights)
        deltas = [numpy.ndarray] * len(weights)
        for j in range(len(weights) - 1, -1, -1):
            if j == len(weights) - 1:
                errors[j] = target - y[j]
                deltas[j] = errors[j]
            else:
                errors[j] = numpy.matmul(weights[j + 1].transpose(), deltas[j + 1])
                deltas[j] = y[j].transpose() * (1 - y[j].transpose()) * errors[j]
        # update weight
        for j in range(0, len(weights), 1):
            if j == 0:
                iGradient = alpha * numpy.matmul(deltas[j], x)
            else:
                iGradient = alpha * numpy.matmul(deltas[j], y[j - 1])
            weights[j] = weights[j] + iGradient.copy()
    return weights


class YoonNeuron:
    # The shared area of YoonDataset class
    # All of instances are using this shared area
    def __init__(self):
        self.input = None
        self.weight = None
        self.output = None

    def load_source(self, file_path: str):
        data_array = numpy.load(file_path)
        self.input = data_array['input']
        self.output = data_array['output']

    def save_result(self, file_path: str):
        numpy.savez(file_path, input=self.input, output=self.output)

    def load_weight(self, file_path: str):
        data_array = numpy.load(file_path)
        self.weight = data_array['weight']

    def save_weight(self, file_path: str):
        numpy.savez(file_path, weight=self.weight)

    def train(self, epoch=1000, scale=0.1, is_init_weight=True, is_run_test=True):
        count, dim = self.input.shape
        input_array = numpy.column_stack((self.input, numpy.ones([count, 1])))
        target_array = self.output
        loss_array = numpy.zeros([epoch, 1])
        if is_init_weight:
            # Weight is random value in -0.1 ~ 0.1 for training
            weights = scale * (2 * numpy.random.random((1, dim + 1)) - 1)
        else:
            weights = self.weight.copy()
        # Train
        for i in range(epoch):
            weights = gradient_descent(array_x=input_array, array_w=weights, array_y=target_array)
            target_estimated = logistic_regression(input_array, weights)
            loss_array[i] = 1 / 2 * numpy.matmul((self.output - target_estimated).transpose(),
                                                 (self.output - target_estimated)) / count
            if i % 100 == 0:
                print("epoch={0}, Error={1:.5f}%".format(i, float(loss_array[i]) * 100))
        self.weight = weights.copy()
        # Test weight
        if is_run_test:
            self.process()

    def process(self, is_save=False):
        count, dim = self.input.shape
        input_transform = numpy.column_stack((self.input, numpy.ones([count, 1])))
        results = (logistic_regression(input_transform, self.weight) >= 0.5) * 1
        print('The accuracy is {:.5f}%'.format(numpy.sum(self.output == results) / count * 100))
        if is_save:
            self.output = results
        return results

    def show_plot(self):
        init_pos1 = numpy.where(self.output == 0)[0]
        init_pos2 = numpy.where(self.output == 1)[0]
        init_pos = (init_pos1, init_pos2)
        colors = ("red", "green")
        for pos, color in zip(init_pos, colors):
            matplotlib.pyplot.scatter(self.input[pos, 0], self.input[pos, 1], alpha=1.0, c=color)
        step = 0.025
        array_x = numpy.arange(0.0, 1.0 + step, step)
        array_y = numpy.arange(0.0, 1.0 + step, step)
        mesh_x, mesh_y = numpy.meshgrid(array_x, array_y)
        mesh_z = numpy.zeros(mesh_x.shape)
        for x in range(mesh_x.shape[0]):
            for y in range(mesh_y.shape[0]):
                mesh_array = numpy.array([mesh_x[x][y], mesh_y[x][y], 1], ndmin=2)
                mesh_z[x][y] = logistic_regression(mesh_array, self.weight)
        matplotlib.pyplot.contour(array_x, array_y, mesh_z, (0.49, 0.51))
        matplotlib.pyplot.show()


class YoonNetwork:
    # The shared area of YoonDataset class
    # All of instances are using this shared area
    def __init__(self):
        self.input = None
        self.weights = None
        self.output = None

    def load_source(self, file_path: str):
        data_array = numpy.load(file_path)
        self.input = data_array['input']
        self.output = data_array['output']

    def save_result(self, file_path: str):
        numpy.savez(file_path, input=self.input, output=self.output)

    def load_weight(self, file_path: str):
        data_array = numpy.load(file_path)
        self.weights = []
        for i in range(len(data_array.files)):
            self.weights.append(data_array['arr_{0}'.format(i)])
        print("The weight order is {0}".format(len(self.weights)))

    def save_weight(self, file_path: str):
        args = (self.weights[i] for i in range(len(self.weights)))
        numpy.savez(file_path, *args)

    def train(self, count=5000, layer_size=10, order=3, output_dim=1, scale=1.0,
              is_init_weight=True, is_run_test=True):
        if count < 1000 or order < 2 or layer_size < 10:
            raise Exception("Train arguments is too little, Epoch: {0}, Size {1}, Order {2}".
                            format(count, layer_size, order))
        data_len, input_dim = self.input.shape
        input_array = numpy.column_stack((self.input, numpy.ones([data_len, 1])))
        target_array = self.output
        loss_array = numpy.zeros([count, 1])
        # Weight is random value in -0.1 ~ 0.1 for training
        if self.weights is not None and not is_init_weight and order == len(self.weights):
            weights = self.weights.copy()
        else:
            # Weight is random value in -0.1 ~ 0.1 for training
            weights = []
            for i in range(order):
                if i == 0:
                    weights.append(scale * (2 * numpy.random.random((layer_size, input_dim + 1)) - 1))
                elif i == order - 1:
                    weights.append(scale * (2 * numpy.random.random((output_dim, layer_size)) - 1))
                else:
                    weights.append(scale * (2 * numpy.random.random((layer_size, layer_size)) - 1))
        # Train
        for i in range(count):
            weights = back_propagation(array_x=input_array, weights=weights, array_y=target_array)
            target_estimated = self.__feed_forward_network(list_x=input_array, list_w=weights)
            loss_array[i] = 1 / 2 * numpy.matmul((self.output - target_estimated).transpose(),
                                                 (self.output - target_estimated)) / data_len
            if i % 100 == 0:
                print("epoch={0}, Error={1:.5f}%".format(i, float(loss_array[i]) * 100))
        self.weights = weights.copy()
        # Test weight
        if is_run_test:
            self.process()

    def process(self, is_save=False):
        data_len, input_dim = self.input.shape
        results = (self.__feed_forward_network() >= 0.5) * 1
        print('The accuracy is {:.5f}%'.format(numpy.sum(self.output == results) / data_len * 100))
        if is_save:
            self.output = results
        return results

    def __feed_forward_network(self, list_x: list = None, list_w: list = None):
        if list_x is None:
            nCountData, nDimInput = self.input.shape
            array_x = numpy.column_stack((self.input, numpy.ones([nCountData, 1])))
        else:
            array_x = numpy.ndarray(list_x)
        if list_w is None:
            list_w = self.weights
        result = array_x
        for j in range(0, len(list_w), 1):
            assert isinstance(list_w[j], numpy.ndarray)
            result = logistic_regression(result.copy(), list_w[j])
        return result

    def show_plot(self):
        init_pos1 = numpy.where(self.output == 0)[0]
        init_pos2 = numpy.where(self.output == 1)[0]
        init_pos = (init_pos1, init_pos2)
        colors = ("red", "green")
        for pos, color in zip(init_pos, colors):
            matplotlib.pyplot.scatter(self.input[pos, 0], self.input[pos, 1], alpha=1.0, c=color)
        step = 0.025
        array_x = numpy.arange(0.0, 1.0 + step, step)
        array_y = numpy.arange(0.0, 1.0 + step, step)
        mesh_x, mesh_y = numpy.meshgrid(array_x, array_y)
        mesh_z = numpy.zeros(mesh_x.shape)
        for x in range(mesh_x.shape[0]):
            for y in range(mesh_y.shape[0]):
                input_array = numpy.array([mesh_x[x][y], mesh_y[x][y], 1], ndmin=2)
                mesh_z[x][y] = self.__feed_forward_network(input_array)
        matplotlib.pyplot.contour(array_x, array_y, mesh_z, (0.49, 0.51))
        matplotlib.pyplot.show()

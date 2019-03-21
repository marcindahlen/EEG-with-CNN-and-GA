import utility
import random


class LstmNeuron(object):
    """
    Class defines a single neuron of long-short term memory type.
    → https://cdn-images-1.medium.com/max/1250/1*laH0_xXEkFE0lKJu54gkFQ.png
    """

    #@TODO konstruktor może, ale nie musi przyjmować zapamiętane wagi
    def __init__(self, window, from_existing_data=False, weights_data=[]):
        self.weights = [[] for i in range(4)]
        self.bias_weights = []
        self.suma_in, self.suma_out, self.suma_mem, self.suma_forget = 0, 0, 0, 0
        self.y_in, self.y_forget, self.state, self.y_out = 0, 0, 0, 0
        self.bias_in, self.bias_out, self.bias_forget, self.bias_mem = 1, 1, 1, 1
        self.mem = 0
        self.output = 0
        if not from_existing_data:
            for j in range(4):
                for i in range(window):
                    self.weights[j].append(1 / random.randint(1, 4 * window))
                self.bias_weights.append(1 / random.randint(1, 4 * window))
            self.waga_prev = 1 / random.randint(1, 4 * window)
        if from_existing_data:
            if not weights_data:
                raise Exception('No weights data passed to the neuron constructor!')
            elif len(weights_data) != 4:
                raise Exception('List passed to the neuron constructor has wrong dimensions!')
            else:
                for j in weights_data:
                    for i in j:
                        self.weights[j].append(weights_data[j][i])
                    self.bias_weights.append(self.weights[j].pop())
                self.waga_prev = weights_data[4].pop()
        self.y_prev = 0

    def get_size(self):
        """
        Returns number of all weights of this neuron.
        :return: int
        """
        size = 0
        for gate in self.weights:
            size += len(gate)
        return size + 5

    def get_weights(self):
        """
        Method outputs neuron's weights in form of lists.
        There is some commotion in saving bias weights and previous value weight.
        :return: a list of four lists
        """
        output = [[j for j in i] for i in self.weights]
        output[len(output)-1].append(self.waga_prev)
        for i, x in enumerate(self.bias_weights):
            output[i].append(x)

        return output

    def get_weights_structured(self):
        """
        Returns neuron's weights in the same order
        as method set_weights() would read them.
        :return: list of four lists
        """
        output = self.weights
        output[3].append(self.waga_prev)
        for i in range(4):
            output[i].append(self.bias_weights[i])
        return output

    def set_weights(self, weights_data=[]):
        """
        Weights should be given as nested list:
        in a list there should be four lists, one for each gate,
        each list's last element should be bias weight,
        and the last list's last but one float should be previous output's weight.
        :param weights_data:
        :return: void
        """
        if not weights_data:
            raise Exception('No weights data passed to the neuron constructor!')
        elif len(weights_data) != 4:
            raise Exception('List passed to the neuron constructor has wrong dimensions!')
        else:
            for j in weights_data:
                for i in j:
                    self.weights[j].append(weights_data[j][i])
                self.bias_weights.append(self.weights[j].pop())
            self.waga_prev = weights_data[4].pop()

    def calculate(self, input=[]):
        """
        Executes a single forward pass on a neuron.
        """
        self.y_prev = self.output
        self.state = self.mem
        # ?? self.stan += self.mem @TODO why?
        self.suma_in = 0
        for i in range(len(input)):
            self.suma_in += input[i] * self.weights[0][i]
        self.suma_in += self.bias_weights[0] * self.bias_in
        self.y_in = utility.sigmoid(self.suma_in)

        self.suma_forget = 0
        for i in range(len(input)):
            self.suma_forget += input[i] * self.weights[1][i]
        self.suma_forget += self.bias_weights[1] * self.bias_forget
        self.y_forget = utility.sigmoid(self.suma_forget)

        self.suma_mem = 0
        for i in range(len(input)):
            self.suma_mem += input[i] * self.weights[2][i]
        self.suma_mem += self.bias_weights[2] * self.bias_mem
        self.suma_mem += self.y_prev * self.waga_prev
        self.mem = self.y_forget * self.state + self.y_in * utility.tanh(self.suma_mem)

        self.suma_out = 0
        for i in range(len(input)):
            self.suma_out += input[i] * self.weights[3][i]
        self.suma_out += self.bias_weights[3] * self.bias_out
        self.y_out = utility.sigmoid(self.suma_out)

        self.output = utility.tanh(self.mem) * self.y_out

        return self.output

    def learn(self, target, learning_lambda, input=[]):
        """
        Executes a single learning pass on a neuron.
        In case of supervised learning.
        (that means target data is known)
        """
        self.d_wagi = [[] for i in range(4)]
        for j in range(4):
            for i in range(len(input)):
                self.d_wagi[j].append(0)

        for i in range(len(input)):
            self.d_wagi[0][i] = target * self.y_out * utility.derivative_tanh(
                self.mem) * self.y_forget * utility.derivative_tanh(
                self.suma_mem) * utility.derivative_sigmoid(self.suma_in) * input[i]
            self.d_wagi[1][i] = target * self.y_out * utility.derivative_tanh(
                self.mem) * self.y_in * utility.derivative_tanh(self.suma_mem) * utility.derivative_sigmoid(
                self.suma_forget) * input[i]
            self.d_wagi[2][i] = target * self.y_out * utility.derivative_tanh(
                self.mem) * self.y_forget * self.y_in * utility.derivative_tanh(self.suma_mem) * input[i]
            self.d_wagi[3][i] = target * utility.derivative_tanh(self.mem) * utility.derivative_sigmoid(
                self.suma_out) * input[i]
        self.bias_weights[0] = learning_lambda * target * self.y_out * utility.derivative_tanh(
            self.mem) * self.y_forget * utility.derivative_tanh(self.suma_mem) * utility.derivative_sigmoid(
            self.suma_in) * self.bias_in
        self.bias_weights[1] = learning_lambda * target * self.y_out * utility.derivative_tanh(
            self.mem) * self.y_in * utility.derivative_tanh(self.suma_mem) * utility.derivative_sigmoid(
            self.suma_forget) * self.bias_out
        self.bias_weights[2] = learning_lambda * target * self.y_out * utility.derivative_tanh(
            self.mem) * self.y_forget * self.y_in * utility.derivative_tanh(self.suma_mem) * self.bias_mem
        self.bias_weights[3] = learning_lambda * target * utility.derivative_tanh(self.mem) * utility.derivative_sigmoid(
            self.suma_out) * self.bias_forget
        self.waga_prev += learning_lambda * target * self.y_out * utility.derivative_tanh(
            self.mem) * self.y_forget * self.y_in * utility.derivative_tanh(self.suma_mem) * self.y_prev

        for i in range(len(input)):
            self.weights[0][i] += learning_lambda * self.d_wagi[0][i]
            self.weights[1][i] += learning_lambda * self.d_wagi[1][i]
            self.weights[2][i] += learning_lambda * self.d_wagi[2][i]
            self.weights[3][i] += learning_lambda * self.d_wagi[3][i]

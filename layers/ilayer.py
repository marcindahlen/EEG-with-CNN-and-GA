class ILayer(object):
    """"""
    def __init__(self, in_shape, out_shape):
        self.output = None
        self.dimensions = in_shape
        self.out_shape = out_shape
        self.weights = []

    def forward_pass(self, input):
        """"""
        pass

    def get_all_weights(self, weights):
        """"""
        pass

    def set_all_weights(self):
        """"""
        pass

    def decompose_weights(self):
        """"""
        pass

    def rebuild_weights(self, flatline_weights):
        """"""
        pass

    def init_weights(self, kernels: int) -> list:
        """"""
        pass


class INetwork(object):
    """"""
    def __init__(self, layer_types, ins_outs_shapes):
        self.layers = None
        self.output = None
        self.score = None
        self.weight_lengths_by_layer = None

    def forward_pass(self, input):
        """"""
        pass

    def initialize(self, layer_types, ins_outs_shapes):
        """"""
        pass

    def evaluate_self(self, target):
        """"""
        pass

    def get_score(self):
        """"""
        pass

    def set_weights(self, weights):
        pass

    def get_weights(self):
        pass

    def save_weights(self):
        pass

    def load_weights(self, filename):
        pass

    def mutate(self):
        """"""
        pass

    def create_single_child(self, other):
        """"""
        pass

    def multiplication_by_budding(self):
        """"""
        pass

    def decompose_weights(self):
        """"""
        pass

    def rebuild_weights(self, flat_weights):
        """"""
        pass

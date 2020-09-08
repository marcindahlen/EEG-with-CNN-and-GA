
class INetwork(object):
    """"""
    def __init__(self, layer_types, ins_outs_shapes):
        self.layers = None
        self.output = None
        self.score = None

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

    def get_id(self):
        """"""
        pass

    def save_state_binary(self):
        """"""
        pass

    def load_state_binary(self):
        """"""
        pass

    def save_state_text(self):
        """"""
        pass

    def load_state_text(self):
        """"""
        pass

    def mutate(self):
        """"""
        pass

    def create_single_child(self):
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

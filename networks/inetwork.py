
class INetwork(object):
    """"""
    def __init__(self):
        self.layers = []

    def forward_pass(self):
        """"""
        pass

    def evaluate_self(self):
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

    def rebuild_weights(self, flatline_weights):
        """"""
        pass

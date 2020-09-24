import numpy


class TestEvolution:
    def test_evo(self):
        sample_length = 100
        samples_no = 100
        domain = [x for x in range(sample_length)]
        data_targets = numpy.random.random(samples_no)
        data_targets = [(x - 0.5) * 2 for x in data_targets]    # [0.0, 1.0] -> [-1.0, 1.0]
        data_targets = numpy.array(data_targets)
        data_source = [[] for n in range(samples_no)]

        for sample_no in range(samples_no):
            data_source[sample_no] = list(map(lambda x: data_targets[sample_no] * x, domain))

        print(data_source[0])

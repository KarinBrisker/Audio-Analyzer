from pipeline import Pipe


class ClapClassifier(Pipe):
    def __init__(self, name):
        super().__init__(name=name)

    def __call__(self, analyzed):
        return analyzed

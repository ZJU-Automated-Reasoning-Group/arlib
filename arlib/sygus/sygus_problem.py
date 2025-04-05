class SygusParser:
    def __init__(self, problem_str):
        self.problem_str = problem_str
        self.problem_str = self.problem_str.replace("\n", "")
        self.problem_str = self.problem_str.replace(" ", "")
        self.problem_str = self.problem_str.replace("(", "")
        self.problem_str = self.problem_str.replace(")", "")

    def parse(self):
        """
        :return:
        """
        tokens = self.problem_str.split(";")
        assert len(tokens) == 2
        assert tokens[0] == "check-synth"
        assert tokens[1].startswith("(= (")
        tokens = tokens[1].split(")")
        assert len(tokens) == 2

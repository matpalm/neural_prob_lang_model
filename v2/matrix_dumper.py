class MatrixDumper(object):

    def __init__(self, filename, matrix, token_idx=None):
        self.output = open(filename, 'w')
        self.matrix = matrix
        self.token_idx = token_idx
        # tab seperated header.
        columns = ["epoch", "batch", "iter", "idx"]
        if token_idx:
            columns.append("label")        
        num_matrix_columns = matrix.get_value().shape[1]
        columns += ["d%i" % d for d in range(num_matrix_columns)]
        self.output.write("\t".join(columns) + "\n")

    def dump(self, e, b, i):
        for idx, embedding in enumerate(self.matrix.get_value()):
            self.output.write("\t".join(map(str, [e, b, i, idx])))
            if self.token_idx:
                self.output.write("\t%s" % self.token_idx.idx_token[idx])
            self.output.write("\t" + "\t".join(map(str, embedding)) + "\n")
        self.output.flush()
        
        

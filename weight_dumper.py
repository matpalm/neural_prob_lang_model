class WeightDumper(object):
    """
    on demand dump a set of layer weights to a file
    """
    def __init__(self, filename, layer, node_names=None, feature_names=None, capture_freq=1):
        self.W = layer.W
        self.num_nodes = layer.W.shape.eval()[1]
        self.b = layer.b if hasattr(layer, 'b') else None
        self.node_names = node_names
        self.feature_names = feature_names
        self.capture_freq = capture_freq
        self.output_f = open(filename, 'w')
        self.output_f.write("epoch\tnode\tfeature\tweight\n")

    def name_for_node(self, idx):
        if self.node_names and idx in self.node_names:
            return self.node_names[idx]
        else:
            return "n%s" % idx

    def name_for_feature(self, idx):
        if self.feature_names and idx in self.feature_names:
            return self.feature_names[idx]
        else:
            return "x%s" % idx

    def take_snapshot(self, epoch):
        if epoch % self.capture_freq != 0:
            return
        for nodeIdx in range(self.num_nodes):
            node_name = self.name_for_node(nodeIdx)
            # dump node weights
            _w = self.W.eval()
            for featureIdx, weight in enumerate(_w[0:,nodeIdx]):
                feature_name = self.name_for_feature(featureIdx)
                self.output_f.write("%s\t%s\t%s\t%s\n" % (epoch, node_name, feature_name, weight))
            # dump bias (for layers that have one)
            if self.b:
                _b = self.b.eval()
                self.output_f.write("%s\t%s\tbias\t%s\n" % (epoch, node_name, _b[nodeIdx]))
            self.output_f.flush()

    def close(self):
        self.output_f.close()
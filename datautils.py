class BatchReader(object):
    """Get batch data recurrently from a file.
    """
    def __init__(self, file_name, max_epoch=None, batch_size=None):
        self.fname = file_name
        self.max_epoch = max_epoch
        self.default_batch_size = batch_size
        self.nepoch = 0
        self.fp = None

    def __del__(self):
        if self.fp is not None:
            self.fp.close()

    def _get_batch(self, batch_size, out):
        if self.fp is None:
            if (not self.max_epoch) or self.nepoch < self.max_epoch:
                # if max_epoch not set or num_epoch not reach the limit
                self.fp = open(self.fname)
                self.nepoch += 1
            else:  # reach max_epoch limit
                return out
        for line in self.fp:
            out.append(line.rstrip('\n'))
            if len(out) >= batch_size:
                break
        else:
            self.fp.close()
            self.fp = None
            return self._get_batch(batch_size, out)
        return out

    def get_batch(self, batch_size=None):
        if batch_size is None:
            batch_size = self.default_batch_size
        return self._get_batch(batch_size, [])

    def __next__(self):
        data = self.get_batch()
        if not data:
            raise StopIteration
        return data

    def __iter__(self):
        return self

    next = __next__


def sequence_input_func(data):
    bs = len(data)
    max_len = 0
    x_idx = []
    x_vals = []
    y_vals = []
    for i, inst in enumerate(data):
        flds = inst.split('\t')
        label = float(flds[0])
        feats = sorted(map(int, flds[1:]))
        if len(feats) > max_len:
            max_len = len(feats)
        for col, feat in enumerate(feats):
            x_idx.append([i, col])
            x_vals.append(feat)
        y_vals.append([label])
    x_shape = [bs, max_len]
    return (x_idx, x_vals, x_shape), y_vals


def index_input_func(data, dim):
    bs = len(data)
    x_idx = []
    x_vals = []
    y_vals = []
    for i, inst in enumerate(data):
        flds = inst.split('\t')
        label = float(flds[0])
        feats = sorted(map(int, flds[1:]))
        for feat in feats:
            x_idx.append([i, feat])
            x_vals.append(1)
        y_vals.append([label])
    x_shape = [bs, dim]
    return (x_idx, x_vals, x_shape), y_vals


def libsvm_input_func(data):
    bs = len(data)
    max_len = 0
    x_idx = []
    x_vals1 = []
    x_vals2 = []
    y_vals = []
    for i, inst in enumerate(data):
        flds = inst.split(' ')
        label = float(flds[0])
        feats = flds[1:]
        if len(feats) > max_len:
            max_len = len(feats)
        for col, feat in enumerate(feats):
            idx, val = feat.split(':')
            idx = int(idx)
            val = float(val)
            x_idx.append([i, col])
            x_vals1.append(feat)
            x_vals2.append(val)
        y_vals.append([label])
    x_shape = [bs, mex_len]
    return (x_idx, x_vals1, x_shape), (x_idx, x_vals2, x_shape), y_vals

from dotmap import DotMap
import sys
import termios
import tty



class AttrDict(DotMap):

    def __getitem__(self, item):
        if isinstance(item, str) and '/' in item:
            item_split = item.split('/')
            curr_item = item_split[0]
            next_item = '/'.join(item_split[1:])
            return self[curr_item][next_item]
        else:
            return super(AttrDict, self).__getitem__(item)

    def __setitem__(self, key, value):
        if isinstance(key, str) and '/' in key:
            key_split = key.split('/')
            curr_key = key_split[0]
            next_key = '/'.join(key_split[1:])
            self[curr_key][next_key] = value
        else:
            super(AttrDict, self).__setitem__(key, value)


    def pprint(self, str_max_len=5):
        str_self = self.leaf_apply(lambda x: str(x)[:str_max_len] + '...')
        return super(AttrDict, str_self).pprint(pformat='json')

    def leaf_keys(self):
        def _get_leaf_keys(d, prefix=''):
            for key, value in d.items():
                new_prefix = prefix + '/' + key if len(prefix) > 0 else key
                if isinstance(value, AttrDict):
                    yield from _get_leaf_keys(value, prefix=new_prefix)
                else:
                    yield new_prefix

        yield from _get_leaf_keys(self)

    def leaf_values(self):
        for key in self.leaf_keys():
            yield self[key]

    def leaf_items(self):
        for key in self.leaf_keys():
            yield key, self[key]

    def leaf_filter(self, func):
        d = AttrDict()
        for key, value in self.leaf_items():
            if func(key, value):
                d[key] = value
        return d

    def leaf_assert(self, func):
        """
        Recursively asserts func on each value
        :param func (lambda): takes in one argument, outputs True/False
        """
        for value in self.leaf_values():
            assert func(value)

    def leaf_modify(self, func):
        """
        Applies func to each value (recursively), modifying in-place
        :param func (lambda): takes in one argument and returns one object
        """
        for key, value in self.leaf_items():
            self[key] = func(value)

    def leaf_apply(self, func):
        """
        Applies func to each value (recursively) and returns a new AttrDict
        :param func (lambda): takes in one argument and returns one object
        :return AttrDict
        """
        d_copy = self.copy()
        d_copy.leaf_modify(func)
        return d_copy

    def combine(self, d_other):
        for k, v in d_other.leaf_items():
            self[k] = v

    def freeze(self):
        frozen = AttrDict(self, _dynamic=False)
        self.__dict__.update(frozen.__dict__)
        return self

    @staticmethod
    def leaf_combine_and_apply(ds, func):
        leaf_keys = tuple(sorted(ds[0].leaf_keys()))
        for d in ds[1:]:
            assert leaf_keys == tuple(sorted(d.leaf_keys()))

        d_combined = AttrDict()
        for k in leaf_keys:
            values = [d[k] for d in ds]
            d_combined[k] = func(values)

        return d_combined

    @staticmethod
    def from_dict(d):
        d_attr = AttrDict()
        for k, v in d.items():
            d_attr[k] = v
        return d_attr


class Getch:
    @staticmethod
    def getch(block=True):
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch

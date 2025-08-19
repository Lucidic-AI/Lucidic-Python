from collections import OrderedDict

class LRUCache(OrderedDict):
    def __init__(self, capacity: int):
        super().__init__()
        self.capacity = capacity

    def __getitem__(self, key):
        if key not in self:
            raise KeyError(key)
        self.move_to_end(key)  # Mark as recently used
        return super().__getitem__(key)

    def __setitem__(self, key, value):
        if key in self:
            self.move_to_end(key)  # Update position
        super().__setitem__(key, value)
        if len(self) > self.capacity:
            self.popitem(last=False)  # Evict least recently used item
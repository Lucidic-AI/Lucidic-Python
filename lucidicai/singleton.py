import threading
from .errors import LucidicNotInitializedError

lai_inst = {}
_singleton_lock = threading.Lock()

def singleton(class_):
    def getinstance(*args, **kwargs):
        # Thread-safe singleton pattern
        with _singleton_lock:
            inst = lai_inst.get(class_)

            # on first access -> no instance yet
            if inst is None:
                # no args/kwargs -> return a NullClient for Client
                if class_.__name__ == 'Client' and not args and not kwargs:
                    inst = NullClient()
                else:
                    inst = class_(*args, **kwargs)
                lai_inst[class_] = inst
                return inst

            # existing instance present
            # if NullClient and now real init args are passed -> upgrade it
            if isinstance(inst, NullClient) and (args or kwargs):
                inst = class_(*args, **kwargs)
                lai_inst[class_] = inst
            return inst
    
    return getinstance

def clear_singletons():
    lai_inst.clear()


class NullClient:
    """
    A no-op client returned when Lucidic has not been initialized.
    All methods are inert and session is None.
    """
    def __init__(self):
        self.initialized = False
        self.session = None
        self.providers = []

    def set_provider(self, *args, **kwargs):
        pass

    def undo_overrides(self):
        pass

    def clear(self):
        pass
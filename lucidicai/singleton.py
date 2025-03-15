from .errors import LucidicNotInitializedError

lai_inst = {}

def singleton(class_):
    def getinstance(*args, **kwargs):
        if class_ not in lai_inst:
            if class_.__name__ == 'Client' and ('lucidic_api_key' not in kwargs or 'agent_id' not in kwargs):
                raise LucidicNotInitializedError()
            lai_inst[class_] = class_(*args, **kwargs)
        return lai_inst[class_]

    return getinstance

def clear_singletons():
    global lai_inst
    lai_inst = {}

lai_inst = {}


def singleton(class_):
    def getinstance(*args, **kwargs):
        if class_ not in lai_inst:
            lai_inst[class_] = class_(*args, **kwargs)
        return lai_inst[class_]

    return getinstance


def conditional_singleton(class_):
    def getinstance(*args, **kwargs):
        use_singleton = kwargs.pop("use_singleton", True)
        if use_singleton:
            if class_ not in lai_inst:
                lai_inst[class_] = class_(*args, **kwargs)
            return lai_inst[class_]
        else:
            return class_(*args, **kwargs)

    return getinstance


def clear_singletons():
    global lai_inst
    lai_inst = {}

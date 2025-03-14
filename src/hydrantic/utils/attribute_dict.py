class AttributeDict(dict):
    """This class implements a dictionary that allows accessing its keys as attributes. Taken from
    https://github.com/LarsKue/lightning-trainable"""

    def __init__(self, **kwargs):
        for key in kwargs.keys():
            if type(kwargs[key]) is dict:
                kwargs[key] = AttributeDict(**kwargs[key])

        super().__init__(**kwargs)

    def __getattribute__(self, item):
        if item not in self:
            return super().__getattribute__(item)

        return self[item]

    def __setattr__(self, key, value):
        if type(value) is dict:
            value = AttributeDict(**value)

        self[key] = value

    def copy(self):
        """Return a copy of the dictionary again as an AttributeDict."""
        return self.__class__(**super().copy())

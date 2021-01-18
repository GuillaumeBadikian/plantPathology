from collections import namedtuple

import yaml


class obj(object):
    def __init__(self, d):
        for a, b in d.items():
            if isinstance(b, (list, tuple)):
               setattr(self, a, [obj(x) if isinstance(x, dict) else x for x in b])
            else:
               setattr(self, a, obj(b) if isinstance(b, dict) else b)


class Config(object):
    class __Config:
        def __init__(self):
            self.confFile = "../config/config.yaml"

        def getConfig(self):
            with open(self.confFile, "r") as ymlfile:
                cfg = yaml.load(ymlfile.read(), Loader=yaml.FullLoader)
            ymlfile.close()
            return cfg

        def increment_n_test(self):
            conf = self.getConfig()
            with open(self.confFile, "w") as ymlfile:
                conf['plantPathology']['n_test'] = conf['plantPathology']['n_test'] + 1
                yaml.dump(conf, ymlfile)
            ymlfile.close()

        def setConfig(self, conf):
            with open(self.confFile, "w") as ymlfile:
                yaml.dump(conf, ymlfile)
            ymlfile.close()
            return self

    instance = None

    def __new__(self):
        if not Config.instance:
            Config.instance = Config.__Config()
        return Config.instance

    def __getattr__(self, attr):
        return getattr(self.instance, attr)

    def __setattr__(self, attr, val):
        return setattr(self.instance, attr, val)

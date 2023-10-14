from lib.train.admin.environment import env_settings


class Settings:
    """ Training settings, e.g. the paths to datasets and networks."""

    def __init__(self, env_num):
        self.set_default(env_num)

    def set_default(self, env_num):
        self.env = env_settings(env_num)
        self.use_gpu = True

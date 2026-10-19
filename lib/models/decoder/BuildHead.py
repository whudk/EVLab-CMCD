from lib.models.decoder.UperHead import UperHead


class BuildHead:

    def __init__(self, configer=None):
        self.configer = configer

    def build_head(self, name=None, **kwargs):
        if name != "uper_head":
            raise ValueError("This extraction supports uper_head only.")
        return UperHead(**kwargs)

import importlib
#VOCODERS = {'HifiGAN': 'modules.vocoders.hifigan.HifiGAN', 'hifigan': 'modules.vocoders.hifigan.HifiGAN'}
VOCODERS = {}

def register_vocoder(cls):
    VOCODERS[cls.__name__.lower()] = cls
    VOCODERS[cls.__name__] = cls
    return cls


def get_vocoder_cls(hparams):
    if hparams['vocoder'] in VOCODERS:
        return VOCODERS[hparams['vocoder']]
    else:
        vocoder_cls = hparams['vocoder']
        pkg = ".".join(vocoder_cls.split(".")[:-1])
        cls_name = vocoder_cls.split(".")[-1]
        vocoder_cls = getattr(importlib.import_module(pkg), cls_name)
        #VOCODERS['hifigan']=vocoder_cls
        return vocoder_cls

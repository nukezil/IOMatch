from .hook import Hook
from semilearn.core.utils import EMA


class EMAHook(Hook):
    def __init__(self) -> None:
        super().__init__()
    
    def before_run(self, algorithm):
        algorithm.ema = EMA(algorithm.model, algorithm.ema_m)
        algorithm.ema.register()
        if algorithm.resume:
            algorithm.ema.load(algorithm.ema_model)

    def after_train_step(self, algorithm):
        if algorithm.ema is not None:
            algorithm.ema.update()
            algorithm.ema_model.load_state_dict(algorithm.model.state_dict())
            algorithm.ema_model.load_state_dict(algorithm.ema.shadow, strict=False)

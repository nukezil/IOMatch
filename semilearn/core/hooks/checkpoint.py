import os

from .hook import Hook


class CheckpointHook(Hook):
    def __init__(self):
        super().__init__()

    def after_warmup_epoch(self, algorithm):
        save_path = os.path.join(algorithm.save_dir, algorithm.save_name)
        if (not algorithm.distributed) or \
                (algorithm.distributed and algorithm.rank % algorithm.ngpus_per_node == 0):
            algorithm.save_model('pretrained_model.pth', save_path)

    def after_train_step(self, algorithm):
        # must be called after evaluation for saving the best
        save_path = os.path.join(algorithm.save_dir, algorithm.save_name)

        # if self.every_n_iters(algorithm, algorithm.num_eval_iter * 20) or algorithm.it == 1024:
        #     if (not algorithm.distributed) or \
        #             (algorithm.distributed and algorithm.rank % algorithm.ngpus_per_node == 0):
        #         algorithm.save_model(f'model_at_{algorithm.it}_step.pth', save_path)

        if self.every_n_iters(algorithm, algorithm.num_eval_iter) or self.is_last_iter(algorithm):
            if (not algorithm.distributed) or \
                    (algorithm.distributed and algorithm.rank % algorithm.ngpus_per_node == 0):
                algorithm.save_model('latest_model.pth', save_path)

                if algorithm.it == algorithm.best_it:
                    algorithm.save_model('model_best.pth', save_path)

from ...trainers.bit_torch import train
import yerbamate, os
from torch.utils.tensorboard import SummaryWriter

env = yerbamate.Environment()

if env.train:

    # Create a logger.
    workers = os.cpu_count()
    logger = SummaryWriter(log_dir=env["results"], comment=env.name+"restart")
    train(
        logdir=env["results"],
        name=env.name,
        model="BiT-M-R50x1",
        dataset="cifar100",
        datadir=env["datadir"],
        examples_per_class=None,
        batch=128,
        batch_split=2,
        workers=workers,
        base_lr=0.0001,
        eval_every=100,
        save=True,
        ologger=logger,
    )

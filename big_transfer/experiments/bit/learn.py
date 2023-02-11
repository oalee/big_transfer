from ...trainers.bit_torch.trainer import train
from ...models.bit_pytorch.models import load_trained_model, get_model_list
from ...data.bit import get_transforms, mini_batch_fewshot
import torchvision as tv, yerbamate, os
from torch.utils.tensorboard import SummaryWriter

# BigTransfer Medium ResNet50 Width 1
model_name = "BiT-M-R50x1" 
# Choose a model form get_model_list that can fit in to your memoery
# Try "BiT-S-R50x1" if this doesn't works for you 

env = yerbamate.Environment()

train_transform, val_transform = get_transforms(img_size=[32, 32])
data_set = tv.datasets.CIFAR10(
    env["datadir"], train=True, download=True, transform=train_transform
)
val_set = tv.datasets.CIFAR10(env["datadir"], train=False, transform=val_transform)

train_set, val_set, train_loader, val_loader = mini_batch_fewshot(
    train_set=data_set,
    valid_set=val_set,
    examples_per_class=2,  # Fewshot disabled
    batch=128,
    batch_split=2,
    workers=os.cpu_count(),  # Auto-val to cpu count
)

imagenet_weight_path = os.path.join(env["weights_path"], f"{model_name}.npz")
model = load_trained_model(
    weight_path=imagenet_weight_path, model_name=model_name, num_classes=10
)
logger = SummaryWriter(log_dir=env["results"], comment=env.name)

if env.train:
    train(
        model=model,
        train_loader=train_loader,
        valid_loader=val_loader,
        train_set_size=len(train_set),
        save=True,
        save_path=os.path.join(env["results"], f"trained_{model_name}.pt"),
        batch_split=2,
        base_lr=0.003,
        eval_every=100,
        tensorboardlogger=logger,
    )

import torch
from resnext import resnet50
from trainer import Trainer
from torch.utils.data import TensorDataset, ConcatDataset

def test_answer():
    net = resnet50()
    trainer = Trainer(net)

    dataset = TensorDataset(
        torch.rand(2, 3, 224, 224),
        torch.tensor([0, 1])
    )
    
    trainer.train(dataset, torch.optim.Adam(net.parameters(), lr=1e-3), torch.nn.CrossEntropyLoss(), epochs=1)

test_answer()
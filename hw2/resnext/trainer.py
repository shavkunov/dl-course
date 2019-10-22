from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import torch

class Trainer():
    def __init__(self, network):
        self.net = network
        self.logger = SummaryWriter("test")


    def train(self, dataset, optim, loss, epochs=50, batch=300):
        loader = DataLoader(dataset, batch_size=batch, shuffle=True)
    
        for epoch in range(epochs):
            for input_data, labels in loader:
                optim.zero_grad()
                outputs = self.net(input_data)

                loss_val = loss(outputs, labels)
                loss_val.backward()
                optim.step()
                
            tr_loss, tr_accuracy = self.avg_results(loader, loss)
            value = str(tr_loss) + " : " + str(tr_accuracy)
            self.logger.add_scalar("train loss", tr_loss, epoch)
            self.logger.add_scalar("train accuracy", tr_accuracy, epoch)


    def avg_results(self, data, loss):
        l = 0.0
        acc = 0.0
        N = len(data)
        for input_data, labels in data:
            outputs = self.net(input_data)

            l += loss(outputs, labels).item()
            acc += (torch.max(outputs, 1)[1] == labels).sum().item() / N


        return l / N, acc / N
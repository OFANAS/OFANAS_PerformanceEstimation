import torch
from torch.nn import Module
from torch.optim import SGD
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from evolution_finder import ArchManager
from accuracy_predictor import AccuracyPredictor
from helper import *

class RMSELoss(Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
    def forward(self,yhat,y):
        return torch.sqrt(self.mse(yhat,y))

class Dataset(Dataset):
    def __init__(self, population, accuracy):
        self.population = population
        self.accuracy = accuracy

    def __len__(self):
        return len(self.population)

    def __getitem__(self, index):
        x = self.population[index]
        y = self.accuracy[index]
        return x, y

def get_sample():
    arch_manager = ArchManager()
    sample = arch_manager.random_sample()
    return sample

def get_population(population_size):
    population = []
    for _ in range(population_size):
        sample = get_sample()
        population.append(sample)
    return population

def get_target_accs(population):
    target_accs = []
    for sample in population:
        e, d, r = sample['e'], sample['d'], sample['r'][0]
        subnet = get_subnet(e, d)
        accuracy = get_accuracy(subnet, r)
        target_accs.append(accuracy)
    return target_accs

def get_train_dataset(population, target_accs):
    params = {'batch_size' : 64,
              'shuffle' : True,
              'num_workers' : 6}
    dataset = Dataset(population, target_accs)
    data_generator = DataLoader(training_set, **params)
    return data_generator 

def train(model, train_dataloader, val_dataloader):
    n_epochs = 100
    criterion = RMSELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    for epoch in range(n_epochs):
        train_loss, val_loss = [], []
        # train
        for inputs, targets in train_dataloader:
            inputs = model.get_feats(inputs)
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
        # validate
        for inputs, targets in val_dataloader:
            inputs = model.get_feats(inputs)
            inputs, targets = inputs.to(device), targets.to(device)
            loss = criterion(outputs, targets)
            val_loss.append(loss.item())
        print ("Epoch:", epoch, "Training Loss: ", np.mean(train_loss),
               "Valid Loss: ", np.mean(val_loss))

def main():
    population_size = 16000
    filename = 'store.pickle'
    model = AccuracyPredictor()
    population = get_population(population_size)
    target_accs = get_target_accs(population)
    write_to_pickle(filename, population, target_accs)
    print("DONEEE!!")
    #train_dataset = get_train_dataset(population, target_accs)
    #train(model, dataset)

if __name__ == '__main__':
    main()

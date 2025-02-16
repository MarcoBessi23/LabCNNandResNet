import torch
import os
import matplotlib.pyplot as plt

def calculate_accuracy(model, test_loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():  
        for inputs, labels in test_loader:
            #inputs, labels = inputs.to(device), labels.to(device)

            
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            total   += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    model.train()
    return accuracy


class Trainer():
    def __init__(self, model, loss, optimizer, epochs):
        self.model = model
        self.loss  = loss
        self.optimizer = optimizer
        self.epochs    = epochs
        self.training_loss     = []
        self.training_accuracy = []
        self.test_accuracy     = []

    def train_model(self, training_loader, test_loader):
        
        for epoch in range(self.epochs):
            print(f'----------------------------EPOCH NUMBER {epoch}-------------------------')
            running_loss = 0
            correct = 0
            total   = 0

            for i, data in enumerate(training_loader):
                print(f'iteration {i+1} of {len(training_loader)}')

                inputs, labels = data
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                l = self.loss(outputs, labels)
                l.backward()
                self.optimizer.step()
                running_loss += l.item()

                #CALCULATE TRAIN ACCURACY
                _, predicted = torch.max(outputs, 1)
                total   += labels.size(0)
                correct += (predicted == labels).sum().item()

                if i % 30 == 29:
                    self.training_loss.append(running_loss / 30)
                    print('  batch {} loss: {}'.format(i + 1, self.training_loss[-1]))
                    running_loss = 0
            
            self.training_accuracy.append(correct/total)
            print('calculating Test Accuracy')
            self.test_accuracy.append(calculate_accuracy(self.model, test_loader))
            print(self.test_accuracy[-1])
        return self.training_loss, self.training_accuracy

    def plot_results(self):
        path_train = os.path.join(os.getcwd(), 'Ex2/results_ex2', 'training_loss.png')
        plt.plot(self.training_loss)
        plt.xlabel('batch_idxs')
        plt.ylabel('training loss')
        plt.savefig(path_train)
        plt.close()

        path_accuracy = os.path.join(os.getcwd(), 'Ex2/results_ex2', 'accuracy.png')
        plt.plot(self.training_accuracy, color= 'blue')
        plt.plot(self.test_accuracy, color = 'red')
        plt.xlabel('epochs')
        plt.ylabel('accuracy')
        plt.savefig(path_accuracy)
        plt.close()


def train_fcn_model(model, train_loader, epochs, optimizer, criterion):

    num_batches = len(train_loader)
    model.train()
    training_loss = []
    for epoch in range(epochs):
        print(f'---------------EPOCH : {epoch+1}----------------------------------------')
        running_loss = 0
        for i, (images, labels) in enumerate(train_loader):

            optimizer.zero_grad()
            multi_logits = model(images)
            loss = criterion(multi_logits, labels)
            running_loss += loss.item()
            loss.backward()
            optimizer.step()

            if i%30 == 0:
                print(f'training loss at iter {i} is {loss.item()}')

        training_loss.append(running_loss/num_batches)
        print(f'training loss at epoch {epoch+1} is {training_loss[-1]}')

    return training_loss

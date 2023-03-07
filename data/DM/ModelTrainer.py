import os
import argparse
import glob as glob
from random import random, randrange
import torch
import torch.nn as nn
from torch.utils.data import Dataset

current_path = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--path", help="path to data", required=False)
args = parser.parse_args()

class TextDataset(Dataset):

    def __init__(self, path=None, for_generator=False):

        if path == None:
            return

        files = glob.glob(os.path.join(path, "transcript_*.txt"))

        intents = []

        for file_name in files:
            file = open(file_name, "r")
            lines = file.readlines()

            if lines[1] != "Status: Complete\n":
                continue

            transcript_started = False

            current_example = []

            for i in range(0, len(lines)):
                if transcript_started:
                    if lines[i][0] == "S" or lines[i][0] == "U":
                        current = lines[i][lines[i].find("["):lines[i].find("]") + 1].replace(" ", "")

                        current = current[1:current.find("(")]

                        if current != "":
                            if lines[i][0] == "S":
                                current_example.append(("S", current))
                            else:
                                current_example.append((lines[i][:2], current))

                    else:
                        intents.append(current_example)
                        current_example = []

                if lines[i] == "-------- Transcript --------\n":
                    i+= 1
                    transcript_started = True

        self.training = []

        max_index = 0

        for i in range(1, len(intents)):
            if len(intents[i]) > len(intents[max_index]):
                max_index = i

        longest_length = len(intents[max_index]) - 1

        if(for_generator):
            inputs = torch.zeros([len(intents), longest_length, 18], dtype=torch.float32).cuda()

            blank_line = torch.zeros(18).cuda()
            blank_line[0] = 1

            for i in range(len(intents)):
                current_input = torch.zeros([longest_length, 18], dtype=torch.float32).cuda()

                for j in range(0, longest_length):

                    if (j >= len(intents[i])):
                        current_input[j] = blank_line.detach().clone()
                    else:
                        current_input[j] = self.to_one_hot(intents[i][j][0], intents[i][j][1], for_generator=True)

                inputs[i] = current_input

            self.training = inputs

        else:

            inputs = torch.zeros([len(intents), longest_length, 17], dtype=torch.float32).cuda()
            labels = torch.full([len(intents), longest_length, 1], -1, dtype=torch.long).cuda()

            for i in range(len(intents)):
                current_input = torch.zeros([longest_length, 17], dtype=torch.float32).cuda()
                current_label = torch.full([longest_length, 1], -1, dtype=torch.long).cuda()
                
                for j in range(0, len(intents[i]) - 1):
                    
                    current_input[j] = self.to_one_hot(intents[i][j][0], intents[i][j][1])
                    current_label[j] = self.to_label(intents[i][j + 1][0], intents[i][j + 1][1])

                inputs[i] = current_input
                labels[i] = current_label

            self.training = (inputs, labels)
                

    def to_one_hot(self, user, input, for_generator=False):

        if(for_generator):
            output = torch.zeros(18)

            match user:
                case "S":
                    output[1] = 1
                case "U1":
                    output[2] = 1
                case "U2":
                    output[3] = 1

            match input:
                case "question":
                    output[4] = 1
                case "options":
                    output[5] = 1
                case "accept-answer":
                    output[6] = 1
                case "confirm-agreement":
                    output[7] = 1
                case "final-answer":
                    output[8] = 1
                case "confirm-final-answer":
                    output[9] = 1
                case "offer-answer":
                    output[10] = 1
                case "offer-to-answer":
                    output[11] = 1
                case "check-answer":
                    output[12] = 1
                case "agreement":
                    output[13] = 1
                case "ask-agreement":
                    output[14] = 1
                case "chit-chat":
                    output[15] = 1
                case "reject-option":
                    output[16] = 1
                case "reject-option-agreement":
                    output[17] = 1

            return output
        else:

            output = torch.zeros(17)

            match user:
                case "S":
                    output[0] = 1
                case "U1":
                    output[1] = 1
                case "U2":
                    output[2] = 1

            match input:
                case "question":
                    output[3] = 1
                case "options":
                    output[4] = 1
                case "accept-answer":
                    output[5] = 1
                case "confirm-agreement":
                    output[6] = 1
                case "final-answer":
                    output[7] = 1
                case "confirm-final-answer":
                    output[8] = 1
                case "offer-answer":
                    output[9] = 1
                case "offer-to-answer":
                    output[10] = 1
                case "check-answer":
                    output[11] = 1
                case "agreement":
                    output[12] = 1
                case "ask-agreement":
                    output[13] = 1
                case "chit-chat":
                    output[14] = 1
                case "reject-option":
                    output[15] = 1
                case "reject-option-agreement":
                    output[16] = 1

            return output
    
    def to_label(self, user, input):
        if user == "S":
            match input:
                case "question":
                    return torch.tensor(1).type(torch.LongTensor)
                case "options":
                    return torch.tensor(2).type(torch.LongTensor)
                case "accept-answer":
                    return torch.tensor(3).type(torch.LongTensor)
                case "confirm-agreement":
                    return torch.tensor(4).type(torch.LongTensor)
        
        return torch.tensor(0).type(torch.LongTensor)

    def to_one_hot_label(self, input):

        output = torch.zeros(5, dtype=torch.float64).cuda()

        output[input.item()] = 1

        return output

    def from_one_hot(self, input, threshhold):

        if(input[0][0].data > threshhold):
            return 0

        return torch.argmax(input[0][1:]).data + 1

    def output_to_input(self, input):
        output = torch.zeros(17)

        output[0] = 1
        output[input + 2] = 1

        return output

    def __len__(self):
        return len(self.training[0])
    
    def __getitem__(self, idx):
        return self.training[0][idx], self.training[1][idx]
    
    def get_dataset(self):
        return self.training
    
class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        #self.RNN = nn.RNN(input_size=17, hidden_size=32, num_layers=3, bidirectional=False, batch_first=True, dropout=0.15)
        self.RNN = nn.LSTM(input_size=17, hidden_size=32, num_layers=2, batch_first=True, dropout=0.15)

        self.Linear = nn.Sequential(
            
            nn.LayerNorm(32),
            #nn.Linear(32, 16),
            #nn.SiLU(),
            #nn.LayerNorm(32),
            #nn.Linear(32, 16),
            #nn.SiLU(),
            #nn.LayerNorm(16),
            #nn.Linear(16, 5),
            #nn.LogSoftmax(dim=1),
            #nn.Softmax(dim=1),
        )

        self.Output1 = nn.Sequential(
            nn.Linear(32, 8),
            nn.LayerNorm(8),
            nn.SiLU(),
            nn.Linear(8, 4),
            #nn.LogSoftmax(dim=1),
            nn.Softmax(dim=1),
        )

        self.Output2 = nn.Sequential(
            nn.Linear(32, 8),
            nn.LayerNorm(8),
            nn.SiLU(),
            nn.Linear(8, 1),
            nn.Sigmoid(),
        )

    def initialise_hidden(self):
        return torch.zeros([2, 32])

    def forward(self, input, hidden_layer, cell_layer=None):
        rnn_out = self.RNN(input, (hidden_layer.detach(), cell_layer.detach()))
        #rnn_out = self.RNN(input, hidden_layer)
        linear_out = self.Linear(rnn_out[0])
        #return linear_out, rnn_out[1]
        out1 = self.Output1(linear_out)
        out2 = self.Output2(linear_out)
        #return torch.cat((out2, out1), 1).cuda(), rnn_out[1]
        return torch.cat((out2, out1), 1).cuda(), rnn_out[1][0], rnn_out[1][1]

class CustomLossFunction():
    def __init__(self, labels, formatter):
        self.loss_function = nn.L1Loss()

        self.formatter = formatter

        self.loss_weights = torch.tensor([0, 0, 0, 0, 0])

        for question in labels:
            for line in question:
                if line.item() == -1:
                    break
                self.loss_weights[line.item()]+= 1

        self.loss_weights = torch.divide(1, self.loss_weights)
        self.loss_weights[1] = 0

        self.loss_weights = torch.sqrt(self.loss_weights)
        self.loss_weights = torch.sqrt(self.loss_weights)

        self.loss_weights = nn.functional.normalize(self.loss_weights, dim=0).cuda()

    def calculate(self, output, expected):
        one_hot = self.formatter.to_one_hot_label(expected)

        difference = torch.subtract(output, one_hot)
        if expected == 0:
            difference[1:] = torch.multiply(difference[1:], 0)
        #difference = torch.multiply(difference, self.loss_weights[expected])
        difference = torch.multiply(difference, difference)

        return torch.mean(difference)

def train(model, epochs, learning_rate, training_data):

    data_count = training_data.__len__()
    validation_count = 20
    input_data, input_labels = training_data.get_dataset()

    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
    schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, "min", factor=0.1, patience=4, verbose=True)
    criterion = CustomLossFunction(input_labels, training_data)

    input_data = input_data.cuda()
    input_labels = input_labels.squeeze(2).cuda()

    swapped_input_data = input_data.detach().clone()

    for i in range(0, swapped_input_data.shape[0]):
        for j in range(0, swapped_input_data.shape[1]):
            if input_data[i][j][1] == 1:
                input_data[i][j][1] = 0
                input_data[i][j][2] = 1
            else:
                if input_data[i][j][2] == 1:
                    input_data[i][j][1] = 1
                    input_data[i][j][2] = 0
    
    for i in range(0, epochs):

        validation_loss = 0
        validation_correct = 0
        validation_total = 0

        permutation_indexes = torch.randperm(input_data.size()[0] - validation_count).type(torch.int64).cuda()

        if random() > 0.5:
            current_batch = torch.index_select(input_data, 0, permutation_indexes).cuda()
        else:
            current_batch = torch.index_select(swapped_input_data, 0, permutation_indexes).cuda()

        current_labels = torch.index_select(input_labels, 0, permutation_indexes).cuda()

        for j in range(0, input_data.shape[0]):

            if j < data_count - validation_count:

                model.train()

                hidden_layer = model.initialise_hidden().cuda()
                cell_layer = model.initialise_hidden().cuda()

                current_loss = 0

                for k in range(0, current_batch[j].shape[0]):

                    if current_labels[j][k].item() == -1:
                        break
                    
                    #output, hidden_layer = model(current_batch[j][k].unsqueeze(0), hidden_layer)
                    output, hidden_layer, cell_layer = model(current_batch[j][k].unsqueeze(0), hidden_layer, cell_layer)

                    current_loss+= criterion.calculate(output, current_labels[j][k].unsqueeze(0))

                    if current_batch[j][k][0] == 1 and ((k + 1) == current_batch[j].shape[0] or current_batch[j][k + 1][0] == 0 or current_labels[j][k + 1].item() == -1):
                        #output, hidden_layer = model(training_data.output_to_input(training_data.from_one_hot(output)).unsqueeze(0).cuda(), hidden_layer)
                        output, hidden_layer, cell_layer = model(training_data.output_to_input(training_data.from_one_hot(output)).unsqueeze(0).cuda(), hidden_layer, cell_layer)
                        current_loss+= criterion.calculate(output, training_data.to_label("", ""))

                if current_loss > 0:
                    optimiser.zero_grad()
                    current_loss.backward()
                    optimiser.step()

            else:

                model.eval()

                hidden_layer = model.initialise_hidden().cuda()
                cell_layer = model.initialise_hidden().cuda()

                for k in range(0, input_data[j].shape[0]):

                    if input_labels[j][k].item() == -1:
                        break

                    #output, hidden_layer = model(input_data[j][k].unsqueeze(0), hidden_layer)
                    output, hidden_layer, cell_layer = model(input_data[j][k].unsqueeze(0), hidden_layer, cell_layer)

                    validation_total+= 1

                    if torch.argmax(output).item() == input_labels[j][k].item():
                        validation_correct+= 1

                    validation_loss+= criterion.calculate(output, input_labels[j][k].unsqueeze(0))

        schedule.step(validation_loss)

        print("Epoch: " + str(i + 1) + "/" + str(epochs) + ", Validation Loss: " + str(validation_loss.item()) + ", Validation Accuracy: " + str((validation_correct * 100) / validation_total) + "%")


def main():

    if args.path == None:
        print("Please add path argument")

    training_data = TextDataset(args.path)

    model = RNN().cuda()

    train(model, 30, 0.0005, training_data)
    #train(model, 60, 0.00005, training_data)

    torch.save(model.state_dict(), os.path.join(current_path, "Model.pth"))


if __name__ == '__main__':
    main()
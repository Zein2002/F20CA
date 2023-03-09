import os
import torch
import ModelTrainer

current_path = os.path.dirname(os.path.abspath(__file__))

class DialogManager():
    def __init__(self):
        self.Model = ModelTrainer.RNN()
        self.Model.eval()
        self.Model.load_state_dict(torch.load(os.path.join(current_path, "Model.pth")))
        self.Formatter = ModelTrainer.TextDataset()

        self.max_lines = 5
        self.threshhold = 0.5

        self.conversations = {}

    def new_conversation(self, conversation_id):
        self.conversations[conversation_id] = Conversation(self)

    def terminate_conversation(self, conversation_id):
        del self.conversations[conversation_id]
 
    def process_line(self, conversation_id, user, intent):
        conv = self.conversations[conversation_id]

        formatted_input = self.Formatter.to_one_hot(user, intent)

        output = []
        probabilities = []
        
        #model_output, conv.hidden = self.Model(formatted_input.unsqueeze(0), conv.hidden)
        model_output, conv.hidden, conv.cell = self.Model(formatted_input.unsqueeze(0), conv.hidden, conv.cell)
        probabilities.append(model_output)
        model_output = self.Formatter.from_one_hot(model_output, self.threshhold)
        output.append(model_output)

        i = 0

        while model_output != 0 and i < self.max_lines:

            #model_output, conv.hidden = self.Model(self.Formatter.output_to_input(model_output).unsqueeze(0), conv.hidden)
            model_output, conv.hidden, conv.cell = self.Model(self.Formatter.output_to_input(model_output).unsqueeze(0), conv.hidden, conv.cell)
            probabilities.append(model_output)
            model_output = self.Formatter.from_one_hot(model_output, self.threshhold)

            if model_output != 0:
                output.append(model_output)
            
            i+= 1



        for i in range(0, len(output)):
            output[i] = self.one_hot_to_text(output[i])

        print(probabilities)

        return output
    
    def init_hidden(self):
        return self.Model.initialise_hidden()
    
    def one_hot_to_text(self, input):
        match input:
            case 0:
                return "do nothing"
            case 1:
                return "question"
            case 2:
                return "options"
            case 3:
                return "accept-answer"
            case 4:
                return "confirm-agreement"

class Conversation():
    def __init__(self, Manager, hidden=None, cell=None):
        self.DM = Manager

        if(hidden == None):
            self.hidden = self.DM.init_hidden()
            self.cell = self.DM.init_hidden()
        else:
            self.cell = cell

#Function for testing the model against manual user input
def main():

    DM = DialogManager()

    DM.new_conversation(1)

    #conv = Conversation(DM)

    #Hard coded start to question
    print(["question"])
    #This should make the model print options
    print(DM.process_line(1, "S", "question"))

    #Main loop
    while True:
        user_input = input()

        user = user_input.split(" ")[0]
        intent = user_input.split(" ")[1]

        print(DM.process_line(1, user, intent))

"""
    Input must be the user (U1 or U2), followed by a space followed by the intent of the user

    Example input (1 line at a time and wait for response):
    U1 offer-to-answer
    U2 agreement
    U1 reject-option
    U1 final-answer

    The system should output one of the following for each line
    do nothing
    question
    options
    accept-answer
    confirm-agreement
"""

if __name__ == '__main__':
    main()
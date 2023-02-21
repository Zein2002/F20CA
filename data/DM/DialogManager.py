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

        self.model_memory = self.Model.initialise_hidden()

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

    def get_next(self, user, user_intent, answer=None):

        formatted_input = self.Formatter.to_one_hot(user, user_intent)

        output = []
        
        model_output, self.model_memory = self.Model(formatted_input.unsqueeze(0), self.model_memory)

        #print(model_output)

        model_output = self.Formatter.from_one_hot(model_output)

        output.append(model_output)

        max_lines = 5

        i = 0

        while model_output != 0 and i < max_lines:

            model_output, new_hidden = self.Model(self.Formatter.output_to_input(model_output).unsqueeze(0), self.model_memory)

            model_output = self.Formatter.from_one_hot(model_output)

            if model_output != 0:
                self.model_memory = new_hidden
                output.append(model_output)
            
            i+= 1

        for i in range(0, len(output)):
            output[i] = self.one_hot_to_text(output[i])

        return output

#Function for testing the model against manual user input
def main():

    DM = DialogManager()

    #Hard coded start to question
    print(["question"])
    #This should make the model print options
    print(DM.get_next("S", "question"))

    #Main loop
    while True:
        user_input = input()

        user = user_input.split(" ")[0]
        intent = user_input.split(" ")[1]

        print(DM.get_next(user, intent))

        pass

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
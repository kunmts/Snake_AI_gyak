import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os


class Linear_QNet(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        
        self.linear1 = nn.Sequential(
            nn.Conv2d(in_channels=input_size, out_channels=64, kernel_size=4, stride=1),
            nn.MaxPool2d(2),
            #nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=1),
            #nn.MaxPool2d(3),
            #nn.ReLU(),
            #nn.Conv2d(in_channels=128, out_channels=64, kernel_size=2, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, output_size),
        )

    def forward(self, x):

        return self.linear1(x)

    def save(self, file_name='Uj_CNN_model.py'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        # (n, x)
#        print(next_state.shape,'0')
#        print(state.shape,'0')
        next_state=torch.unsqueeze(next_state,0)
        reward=torch.unsqueeze(reward,0)
        done = (done, )

        # 1: predicted Q values with current state
        pred = self.model(state)

        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if done[idx]==False:
                
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))#.item()
                target[idx][torch.argmax(action[idx]).item()] = torch.sum(Q_new)

            else:
                target[idx][torch.argmax(action[idx]).item()] = torch.sum(Q_new)
 
    
        # 2: Q_new = r + y * max(next_predicted Q value) -> only do this if not done
        # pred.clone()
        # preds[argmax(action)] = Q_new
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()

        self.optimizer.step()

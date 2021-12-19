import os
import random
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image

from game.flappy_bird import GameState

class DQN(nn.Module):
    def __init__(self, in_channels, num_actions):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)

        self.fc1 = nn.Linear(in_features=7*7*64, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=num_actions)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class Dueling_DQN(nn.Module):
    def __init__(self, in_channels, num_actions):
        super(Dueling_DQN, self).__init__()
        self.num_actions = num_actions

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)

        self.fc1_adv = nn.Linear(in_features=7 * 7 * 64, out_features=512)
        self.fc1_val = nn.Linear(in_features=7 * 7 * 64, out_features=512)

        self.fc2_adv = nn.Linear(in_features=512, out_features=num_actions)
        self.fc2_val = nn.Linear(in_features=512, out_features=1)

        self.relu = nn.ReLU()

    def forward(self, x):
        batch_size = x.size(0)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = x.view(x.size(0), -1)

        adv = self.relu(self.fc1_adv(x))
        val = self.relu(self.fc1_val(x))

        adv = self.fc2_adv(adv)
        val = self.fc2_val(val).expand(x.size(0), self.num_actions)

        x = val + adv - adv.mean(1).unsqueeze(1).expand(x.size(0), self.num_actions)
        return x

def init_weights(m):
    if type(m) == nn.Conv2d or type(m) == nn.Linear:
        torch.nn.init.uniform_(m.weight, -0.01, 0.01)
        m.bias.data.fill_(0.01)

def image_to_tensor(image):
    image_tensor = image.transpose(2, 0, 1)
    image_tensor = image_tensor.astype(np.float32)
    image_tensor = torch.from_numpy(image_tensor)
    if torch.cuda.is_available():
        image_tensor = image_tensor.cuda()
    return image_tensor

def resize_and_bgr2gray(image):
    image = image[0:288, 0:404]
    image_data = cv2.cvtColor(cv2.resize(image, (84, 84)), cv2.COLOR_BGR2GRAY)
    image_data[image_data > 0] = 255
    image_data = np.reshape(image_data, (84, 84, 1))
    return image_data


def train(model, model_target, start, double_dqn):
    game_state = GameState()
    number_of_actions = 2
    final_epsilon = 0.0001
    initial_epsilon = 0.1
    number_of_iterations = 2000000
    replay_memory_size = 10000
    minibatch_size = 32
    gamma = 0.99

    optimizer = optim.Adam(model.parameters(), lr=1e-6)
    criterion = nn.MSELoss()

    # initialize replay memory
    replay_memory = []

    # initial action is do nothing
    action = torch.zeros([number_of_actions], dtype=torch.float32)
    action[0] = 1
    image_data, reward, terminal = game_state.frame_step(action)
    image_data = resize_and_bgr2gray(image_data)
    image_data = image_to_tensor(image_data)
    state = torch.cat((image_data, image_data, image_data, image_data)).unsqueeze(0)  # torch.Size([1, 4, 84, 84])

    # initialize epsilon value
    epsilon = initial_epsilon
    iteration = 0
    epsilon_decrements = np.linspace(initial_epsilon, final_epsilon, number_of_iterations)

    # main infinite loop
    while iteration < number_of_iterations:
        # get output from the neural network
        output = model(state)[0]

        # action
        action = torch.zeros([number_of_actions], dtype=torch.float32).cuda()
        # epsilon greedy exploration
        random_action = random.random() <= epsilon
        action_index = [torch.randint(number_of_actions, torch.Size([]), dtype=torch.int)
                        if random_action
                        else torch.argmax(output)][0].cuda()  # 0 or 1
        action[action_index] = 1

        # get next state and reward
        image_data_1, reward, terminal = game_state.frame_step(action)
        image_data_1 = resize_and_bgr2gray(image_data_1)
        image_data_1 = image_to_tensor(image_data_1)
        state_1 = torch.cat((state.squeeze(0)[1:, :, :], image_data_1)).unsqueeze(0)

        action = action.unsqueeze(0)
        reward = torch.from_numpy(np.array([reward], dtype=np.float32)).unsqueeze(0)

        # save transition to replay memory, if replay memory is full, remove the oldest transition
        replay_memory.append((state, action, reward, state_1, terminal))
        if len(replay_memory) > replay_memory_size:
            replay_memory.pop(0)

        # epsilon annealing
        epsilon = epsilon_decrements[iteration]

        # sample random minibatch
        minibatch = random.sample(replay_memory, min(len(replay_memory), minibatch_size))
        state_batch = torch.cat(tuple(d[0] for d in minibatch)).cuda()
        action_batch = torch.cat(tuple(d[1] for d in minibatch)).cuda()
        reward_batch = torch.cat(tuple(d[2] for d in minibatch)).cuda()
        state_1_batch = torch.cat(tuple(d[3] for d in minibatch)).cuda()

        # get output for the next state, set y_j to r_j for terminal state, otherwise to r_j + gamma*max(Q)
        if double_dqn == 0:
            output_1_batch = model_target(state_1_batch)
            y_batch = torch.cat(tuple(reward_batch[i] if minibatch[i][4]
                                      else reward_batch[i] + gamma * torch.max(output_1_batch[i])
                                      for i in range(len(minibatch))))
        else:
            output_inner = model(state_1_batch).detach()
            _, a_index = output_inner.max(1)
            output_1_batch = model_target(state_1_batch)
            y_batch = torch.cat(tuple(reward_batch[i] if minibatch[i][4]
                                      else reward_batch[i] + gamma * output_1_batch[i][a_index[i]]
                                      for i in range(len(minibatch))))


        # extract Q-value
        q_value = torch.sum(model(state_batch) * action_batch, dim=1)

        # PyTorch accumulates gradients by default, so they need to be reset in each pass
        optimizer.zero_grad()
        y_batch = y_batch.detach()
        loss = criterion(q_value, y_batch)
        loss.backward()
        optimizer.step()

        # set state to be state_1
        state = state_1
        iteration += 1

        if iteration % 10000 == 0:
            # torch.save(model, "pretrained_model/current_model_" + str(iteration) + ".pth")
            model_target.load_state_dict(model.state_dict())


            print("iteration:", iteration, "elapsed time:", time.time() - start, "epsilon:", epsilon, "action:",
                  action_index.cpu().detach().numpy(), "reward:", reward.numpy()[0][0], "Q max:",
                  np.max(output.cpu().detach().numpy()))

            torch.save(model, "pretrained_model/current_model" + ".pth")


def test(model):
    game_state = GameState()
    number_of_actions = 2

    # initial action is do nothing
    action = torch.zeros([number_of_actions], dtype=torch.float32)
    action[0] = 1
    image_data, reward, terminal = game_state.frame_step(action)
    image_data = resize_and_bgr2gray(image_data)
    image_data = image_to_tensor(image_data)
    state = torch.cat((image_data, image_data, image_data, image_data)).unsqueeze(0)

    while True:
        # get output from the neural network
        output = model(state)[0]

        action = torch.zeros([number_of_actions], dtype=torch.float32).cuda()

        # get action
        action_index = torch.argmax(output).cuda()
        action[action_index] = 1

        # get next state
        image_data_1, reward, terminal = game_state.frame_step(action)
        image_data_1 = resize_and_bgr2gray(image_data_1)
        image_data_1 = image_to_tensor(image_data_1)
        state_1 = torch.cat((state.squeeze(0)[1:, :, :], image_data_1)).unsqueeze(0)

        # set state to be state_1
        state = state_1

        if terminal:
            print('Game over!')
            print('score:',reward)
            break


def main(mode):
    cuda_is_available = torch.cuda.is_available()

    if mode == 'test':
        model = torch.load(
            'pretrained_model/current_model.pth',
            map_location='cpu' if not cuda_is_available else None
        ).eval().cuda()

        test(model)

    elif mode == 'train':
        if not os.path.exists('pretrained_model/'):
            os.mkdir('pretrained_model/')
        Dueling = 1
        if Dueling == 0:
            model = DQN(in_channels=4, num_actions=2).cuda()
            model_target = DQN(in_channels=4, num_actions=2).cuda()
        else:
            model = Dueling_DQN(in_channels=4, num_actions=2).cuda()
            model_target = Dueling_DQN(in_channels=4, num_actions=2).cuda()
        model.apply(init_weights)
        model_target.apply(init_weights)

        start = time.time()
        double_dqn = 1
        train(model, model_target, start, double_dqn)


if __name__ == "__main__":
    main('test')
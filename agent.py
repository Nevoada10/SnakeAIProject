import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:

    def __init__(self):
        self.n_games = 0  # Number of the games.
        self.epsilon = 0  # A parameter to control de randomness.
        self.gamma = 0.9  # Discount rate.
        self.memory = deque(maxlen=MAX_MEMORY)  # If we exceed the memory we will popleft() elements.
        self.model = Linear_QNet(11, 128, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma, )


    def get_state(self, game):
        head = game.snake[0]  # Gets the snake's head coordinates.
        point_l = Point(head.x - 20, head.y)  # Checks point at left.
        point_r = Point(head.x + 20, head.y)  # Checks point at right.
        point_u = Point(head.x, head.y - 20)  # Checks point at up.
        point_d = Point(head.x, head.y + 20)  # Checks point at down.

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight.
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            # Danger right.
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),

            # Danger left.
            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),

            # Move direction.
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Store food location.
            game.food.x < game.head.x,  # Food is to the left.
            game.food.x > game.head.x,  # Food is to the right.
            game.food.y < game.head.y,  # Food is up.
            game.food.y > game.head.y,  # Food is down.
            ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))  # popleft if MAX_MEMORY is used.

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)  # Returns a list of tuples.
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        #
        #

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # Random Moves %: Called tradeoff ( exploration / exploitation ) in Deep learning.
        self.epsilon = 80 - self.n_games  # Changeable
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move


def train():
    plot_scores = []  # Keep in track the scores.
    plot_mean_scores = []  # Keep in track the average score:
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()
    while True:
        # Get the old state.
        state_old = agent.get_state(game)

        # Choose a move.
        final_move = agent.get_action(state_old)

        # Performing a move abd get a new state.
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # Training short memory.
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # Remember.
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # Train long memory/"replayed" memory. TRAINS again on all the previous games. # Plot the results.
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print("Game", agent.n_games, "Score", score, "Record", record)
            # Plotting.
            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)


if __name__ == "__main__":
    train()

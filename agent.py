import torch
import random
import numpy as np
from collections import deque
from model import DQN, QTrainer
import os
import json

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0  # Exploration
        self.gamma = 0.9  # Discount rate
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = DQN(11, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game, snake):
        """Construit le vecteur d'état de 11 features"""
        head_x = snake.get_x()
        head_y = snake.get_y()
        block_size = snake.get_block_size()

        # Points autour de la tête
        point_l = (head_x - block_size, head_y)
        point_r = (head_x + block_size, head_y)
        point_u = (head_x, head_y - block_size)
        point_d = (head_x, head_y + block_size)

        # Direction actuelle
        dir_l = snake.get_x_velocity() == -block_size
        dir_r = snake.get_x_velocity() == block_size
        dir_u = snake.get_y_velocity() == -block_size
        dir_d = snake.get_y_velocity() == block_size
    

        # Fonction helper pour tester collision
        def is_collision(x, y):
            # Mur
            if x < 0 or x >= game.get_width() or y < 0 or y >= game.get_height():
                return True
            # Queue (utilise position_history de snake)
            if hasattr(snake, '_Snake__position_history'):
                for pos in snake._Snake__position_history[:-1]:
                    if pos == (x, y):
                        return True
            return False

        # Dangers
        danger_straight = (
            (dir_r and is_collision(*point_r)) or
            (dir_l and is_collision(*point_l)) or
            (dir_u and is_collision(*point_u)) or
            (dir_d and is_collision(*point_d))
        )

        danger_right = (
            (dir_u and is_collision(*point_r)) or
            (dir_d and is_collision(*point_l)) or
            (dir_l and is_collision(*point_u)) or
            (dir_r and is_collision(*point_d))
        )

        danger_left = (
            (dir_u and is_collision(*point_l)) or
            (dir_d and is_collision(*point_r)) or
            (dir_l and is_collision(*point_d)) or
            (dir_r and is_collision(*point_u))
        )

        # Position nourriture
        food_x = snake._Snake__food_x
        food_y = snake._Snake__food_y

        state = np.array([
            int(danger_straight),
            int(danger_right),
            int(danger_left),
            int(dir_l),
            int(dir_r),
            int(dir_u),
            int(dir_d),
            int(food_x < head_x),  # nourriture à gauche
            int(food_x > head_x),  # nourriture à droite
            int(food_y < head_y),  # nourriture en haut
            int(food_y > head_y),  # nourriture en bas
        ], dtype=int)

        return state
    def save_training_state(self, file_name="train_state.json"):
        folder = "./model"
        if not os.path.exists(folder):
            os.makedirs(folder)
        path = os.path.join(folder, file_name)
        data = {
            "n_games": self.n_games
        }
        with open(path, "w") as f:
            json.dump(data, f)

    def load_training_state(self, file_name="train_state.json"):
        folder = "./model"
        path = os.path.join(folder, file_name)
        if os.path.exists(path):
            with open(path, "r") as f:
                data = json.load(f)
            self.n_games = data.get("n_games", 0)


    def remember(self, state, action, reward, next_state, done):
        """Stocke une transition dans la mémoire"""
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        """Entraîne sur un batch aléatoire du replay buffer"""
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        """Entraîne sur une seule transition (online)"""
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        """Epsilon-greedy: exploration vs exploitation"""
        # Exploration rate décroît avec le nombre de parties
        self.epsilon = self.epsilon = max(0.01, 1 / (1 + 0.01 * self.n_games))
        final_move = [0, 0, 0]

        if random.random() < self.epsilon:
            # Exploration: action aléatoire
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            # Exploitation: argmax Q(s, a)
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move

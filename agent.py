import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot
import os

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001


class Agent:

    def __init__(self, load_existing_model=False):
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = Linear_QNet(11, 256, 3)

        # Si load_existing_model est True et que le fichier existe, charger le modèle
        if load_existing_model and os.path.exists('./model/model.pth'):
            print("Chargement du modèle existant...")
            self.epsilon
            self.model.load_state_dict(torch.load('./model/model.pth'))
        else:
            print("Aucun modèle existant trouvé, entraînement d'un nouveau modèle...")

        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),
            
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Food location 
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y  # food down
            ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        #for state, action, reward, nexrt_state, done in mini_sample:
        #    self.trainer.train_step(state, action, reward, next_state, done)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 80 - self.n_games
        final_move = [0,0,0]
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
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0

    # Demande à l'utilisateur s'il veut charger un modèle existant
    load_existing_model = input("Voulez-vous charger un modèle existant (model.pth) ? (oui/non) ").strip().lower() == 'oui'

    # Initialiser l'agent avec ou sans chargement du modèle existant
    agent = Agent(load_existing_model=load_existing_model)

    game = SnakeGameAI()
    while True:
        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train long memory, plot result
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print('Game', agent.n_games, 'Score', score, 'Record:', record)

def test():
    # Charger le modèle existant
    load_existing_model = True  # Toujours charger le modèle pré-entraîné
    agent = Agent(load_existing_model=load_existing_model)
    # Créer une instance du jeu
    game = SnakeGameAI()

    # Initialiser les variables de score
    total_score = 0
    record = 0

    while True:
        # Obtenir l'état actuel du jeu
        state_old = agent.get_state(game)

        # Décider de l'action à effectuer avec le modèle (pas d'exploration, juste de l'exploitation)
        final_move = agent.get_action(state_old)

        # Effectuer le mouvement et obtenir les nouveaux résultats du jeu
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # Afficher l'état du jeu à chaque étape si nécessaire (facultatif)
        # game.render()

        # Vérifier si la partie est terminée
        if done:
            # Si la partie est terminée, afficher le score
            print(f"Partie terminée. Score final: {score}, Record: {record}")

            # Garder une trace du meilleur score
            if score > record:
                record = score

            # Réinitialiser le jeu pour une nouvelle partie (si souhaité)
            game.reset()
            total_score += score
            break  # Sortir de la boucle après une partie terminée, ou continuer pour de nouvelles parties

    # Afficher les résultats finaux
    print(f"Score total: {total_score}, Record: {record}")




if __name__ == '__main__':
    choix = input("Voulez-vous entrainer le modèle ? (oui/non) ").strip().lower() == 'oui';
    if choix:
        train()
    else :
        test()

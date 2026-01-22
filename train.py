import pygame
import numpy as np
from agent import Agent
from snake_game import SnakeGame, Snake

def train():
    pygame.init()
    
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    
    game = SnakeGame(400, 400,ticking_speed=0)
    snake = Snake(game.get_width(), game.get_height(), 5, game)
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        state_old = agent.get_state(game, snake)
        
        final_move = agent.get_action(state_old)
        
        reward, done, score = snake.play_step_rl(final_move,game)
        
        state_new = agent.get_state(game, snake)
        
        agent.train_short_memory(state_old, final_move, reward, state_new, done)
        
        agent.remember(state_old, final_move, reward, state_new, done)

        game.get_screen().fill((0, 0, 0))
        snake.draw(game)
        pygame.display.flip()
        game.get_clock().tick(60)

        if done:
            snake.reset()
            agent.n_games += 1
            agent.train_long_memory()
            
            if score > record:
                record = score
                agent.model.save()
                agent.save_training_state()
            
            print(f'Game {agent.n_games}, Score: {score}, Record: {record}')
            
            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)

    pygame.quit()

if __name__ == '__main__':
    train()

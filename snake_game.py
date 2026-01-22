import pygame
import numpy as np

class SnakeGame:
    def __init__(self, width=400, height=400,ticking_speed=10):
        self.__width = width
        self.__height = height
        self.__screen = pygame.display.set_mode((self.__width, self.__height))
        self.__clock = pygame.time.Clock()
        self.__ticking_speed = ticking_speed
    def get_screen(self):
        return self.__screen
    def get_clock(self):
        return self.__clock
    def keys_pressed(self):
        return pygame.key.get_pressed()
    def play (self,screen = None):
        snake.spawn()
        while pygame.event.get() != pygame.QUIT:
            snake.update()
            snake.draw(screen)
            pygame.display.update()
            self.__clock.tick(self.__ticking_speed)
    def get_width(self):
        return self.__width
    def get_height(self):
        return self.__height
class Snake:
    def __init__(self, width, height, block_size,game):
        self.__width = width
        self.__height = height
        self.__block_size = block_size
        self.__x = self.__width // 2
        self.__y = self.__height // 2
        self.__length = 1
        self.__position_history = [(self.__x, self.__y)]
        self.__x_velocity = 0
        self.__y_velocity = -5
        self.__score = 0
        self.__food_x , self.__food_y = self.spawn(game)
        self.__last_action = [-1]
    def play_step_rl(self, action,game):
        if isinstance(action, list) or isinstance(action, np.ndarray):
            action_idx = np.argmax(action)
        else:
            action_idx = action
            
        self.move(action_idx) 
        
        self.displace()

        reward = 0
        done = False
        
        

        if self.collision() or self.tail_collision():
            done = True
            reward = -20 
            return reward, done, self.__score
            

        elif self.food_collision(self.__food_x, self.__food_y):
            self.grow()
            if self.__length == self.__width * self.__height :
                reward = 5000
            else :
                self.__food_x, self.__food_y = self.spawn(game)
                reward = 10 
        else:
            reward = -0.01
        return reward, done, self.__score

    def displace(self):
        self.__x += self.__x_velocity
        self.__y += self.__y_velocity
        self.__position_history.append((self.__x, self.__y))
        if len(self.__position_history) > self.__length:
            self.__position_history.pop(0)

    def grow(self):
        self.__length += 1
        self.__score += 1
    def get_action (self):
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            return 1
        elif keys[pygame.K_RIGHT]:
            return 0
        elif keys[pygame.K_ESCAPE] or keys[pygame.K_q]:
            pygame.quit()
        else :
            return -1
    def move(self,action):
        self.update_last_action(action)
        if action == -1:
            pass        
        elif action == 0 and self.__x_velocity == 5 and self.__y_velocity == 0:
            self.__x_velocity = 0
            self.__y_velocity = 5
        elif action == 0 and self.__x_velocity == -5 and self.__y_velocity == 0:
            self.__x_velocity = 0
            self.__y_velocity = -5
        elif action == 0 and self.__x_velocity == 0 and self.__y_velocity == 5:
            self.__x_velocity = -5
            self.__y_velocity = 0
        elif action == 0 and self.__x_velocity == 0 and self.__y_velocity == -5:
            self.__x_velocity = 5
            self.__y_velocity = 0
        elif action == 1 and self.__x_velocity == 5 and self.__y_velocity == 0:
            self.__x_velocity = 0
            self.__y_velocity = -5
        elif action == 1 and self.__x_velocity == -5 and self.__y_velocity == 0:
            self.__x_velocity = 0
            self.__y_velocity = 5
        elif action == 1 and self.__x_velocity == 0 and self.__y_velocity == 5:
            self.__x_velocity = 5
            self.__y_velocity = 0
        elif action == 1 and self.__x_velocity == 0 and self.__y_velocity == -5:
            self.__x_velocity = -5
            self.__y_velocity = 0
    def reset(self):
        self.__y = self.__height // 2
        self.__x = self.__width // 2
        self.__length = 1
        self.__x_velocity = 0
        self.__y_velocity = -5
        self.__position_history = [(self.__x, self.__y)]
        self.__score = 0
        self.__last_action = [-1]
    def collision(self):
        if self.__x < 0 or self.__x >= self.__width or self.__y < 0 or self.__y >= self.__height:
            return True
        return False
    def food_collision(self, food_x, food_y):
        if self.__x == food_x and self.__y == food_y:
            return True
        return False
    def update(self):
        self.move()
        self.displace()
        
        if self.collision() or self.tail_collision():
            self.reset()
        
        if self.food_collision(self.__food_x, self.__food_y):
            self.grow()
            self.__food_x, self.__food_y = self.spawn()

    def update_last_action(self, action):
        self.__last_action.append(action)
        self.__last_action = self.__last_action[len(self.__last_action)-self.__length:]
    
    def draw(self, screen):
        screen.get_screen().fill((0, 0, 0))
        
        pygame.draw.rect(
            screen.get_screen(),
            (0, 255, 0),
            (self.__food_x, self.__food_y, self.__block_size, self.__block_size)
        )
        
        for i, (x, y) in enumerate(self.__position_history):
            color = (255, 0, 0) if i == len(self.__position_history) - 1 else (200, 0, 0)
            pygame.draw.rect(
                screen.get_screen(),
                color,
                (x, y, self.__block_size, self.__block_size)
            )
    def tail_collision(self):
        for i in range(len(self.__position_history) - 1):
            if self.__x == self.__position_history[i][0] and self.__y == self.__position_history[i][1]:
                return True
        return False

    def spawn(self, game):
        while True:
            food_x = np.random.randint(0, game.get_width() // self.get_block_size()) * self.get_block_size()
            food_y = np.random.randint(0, game.get_height() // self.get_block_size()) * self.get_block_size()
            if (food_x, food_y) not in self._Snake__position_history:
                return food_x,food_y

    def get_state(self, food_x, food_y):
        return np.array([self.__x - food_x, self.__y - food_y, self.__x_velocity, self.__y_velocity])
    def get_score(self):
        return self.__score
    def get_length(self):
        return self.__length
    def get_x(self):
        return self.__x
    def get_y(self):
        return self.__y
    def get_food_x(self):
        return self.__food_x
    def get_food_y(self):
        return self.__food_y
    def get_x_velocity(self):
        return self.__x_velocity
    def get_y_velocity(self):
        return self.__y_velocity
    def get_width(self):
        return self.__width
    def get_height(self):
        return self.__height
    def get_block_size(self):
        return self.__block_size
if __name__ == '__main__':
    game = SnakeGame(800,800,10)
    snake = Snake(game.get_width(), game.get_height(), 5)
    game.play(game)
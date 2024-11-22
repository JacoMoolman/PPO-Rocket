import pygame
import math
import random

class MoonLanderGame:
    def __init__(self, net):
        self.net = net
        self.initialize_game()
        self.steps = 0
        self.total_rewards = 0
        self.stationary_time = 0  # Track how long rocket has been stationary

    def initialize_game(self):
        # Initialize Pygame
        pygame.init()

        # Set up the display
        self.WIDTH = 800
        self.HEIGHT = 600
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption("Moon Lander")
        self.clock = pygame.time.Clock()

        # Clock tick rate
        self.CLOCK_SPEED = 60  # Reduced for better visualization

        # Maximum run time (in game seconds)
        # self.MAX_RUN_TIME = 1000  # Reduced from 1000 to make it more reasonable

        # Load and resize the rocket image
        self.rocket_img = pygame.image.load("Rocket.png")
        self.rocket_img = pygame.transform.scale(self.rocket_img, (self.rocket_img.get_width() // 2, self.rocket_img.get_height() // 2))
        self.rocket_rect = self.rocket_img.get_rect()

        # Load and resize the flames image
        self.flames_img = pygame.image.load("Flames.png")
        self.flames_img = pygame.transform.scale(self.flames_img, (self.rocket_img.get_width(), self.rocket_img.get_height()))
        self.flames_rect = self.flames_img.get_rect()

        # Load and resize the moon image
        self.moon_img = pygame.image.load("Moonimage.png")
        self.target_radius = 20
        self.moon_img = pygame.transform.scale(self.moon_img, (self.target_radius * 2, self.target_radius * 2))
        self.moon_rect = self.moon_img.get_rect()

        # Physics constants
        self.GRAVITY = 0.05  # Reduced from 0.1 to make it easier to control
        self.THRUST = 0.5   # Reduced from 0.2 to make it easier to control
        self.ROTATION_SPEED = 3
        self.MAX_SPEED = 5
        self.MAX_ANGULAR_VELOCITY = 10

        # Initialize font
        pygame.font.init()
        self.font = pygame.font.Font(None, 36)

        # Generate random stars
        self.stars = []
        for _ in range(100):
            x = random.randint(0, self.WIDTH)
            y = random.randint(0, self.HEIGHT)
            size = random.randint(1, 3)
            self.stars.append((x, y, size))

        self.reset_game()

    def reset_game(self):
        # Reset game state
        self.position = pygame.math.Vector2(self.WIDTH // 2, self.HEIGHT // 4)
        self.velocity = pygame.math.Vector2(0, 0)
        self.angle = 0
        self.angular_velocity = 0
        self.prev_angle = 0
        self.running = True
        self.thrust = False
        self.target_pos = None

        self.rocket_rect.center = self.position

        self.generate_target_position()

        # Calculate the initial distance between the rocket and the moon
        self.initial_distance = self.position.distance_to(self.target_pos)

        # Reset timer
        self.timer = 0

        # Reset zero speed time, zero X movement time, and penalty flag
        self.zero_speed_time = 0
        self.zero_x_movement_time = 0
        self.penalty_applied = False

    def generate_target_position(self):
        while True:
            x = random.randint(self.target_radius, self.WIDTH - self.target_radius)
            y = random.randint(self.target_radius, self.HEIGHT - self.target_radius)
            self.target_pos = pygame.math.Vector2(x, y)
            self.moon_rect.center = self.target_pos
            if self.target_pos.distance_to(self.position) > 100:
                break

    def run_step(self):
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return None, 0, True  # Signal game end

        # Calculate inputs for neural network
        x_distance = (self.target_pos.x - self.position.x) / self.WIDTH
        y_distance = (self.target_pos.y - self.position.y) / self.HEIGHT
        angle_to_moon = math.degrees(math.atan2(y_distance, x_distance))
        normalized_angle = angle_to_moon / 180

        elapsed_time = self.clock.get_time() / 1000
        if elapsed_time > 0:
            angular_velocity = (self.angle - self.prev_angle) / elapsed_time
            normalized_angular_velocity = angular_velocity / self.MAX_ANGULAR_VELOCITY
        else:
            normalized_angular_velocity = 0

        relative_velocity_x = self.velocity.x / self.MAX_SPEED
        relative_velocity_y = self.velocity.y / self.MAX_SPEED
        distance_to_vertical_edge = min(self.position.x, self.WIDTH - self.position.x) / self.WIDTH
        distance_to_horizontal_edge = min(self.position.y, self.HEIGHT - self.position.y) / self.HEIGHT
        angle_between_direction_velocity = math.atan2(self.velocity.y, self.velocity.x) - math.radians(self.angle)

        state = [
            self.position.x / self.WIDTH,
            self.position.y / self.HEIGHT,
            self.angle / 360,
            self.velocity.x / self.MAX_SPEED,
            self.velocity.y / self.MAX_SPEED,
            self.position.distance_to(self.target_pos) / math.sqrt(self.WIDTH**2 + self.HEIGHT**2),
            x_distance,
            y_distance,
            normalized_angle,
            normalized_angular_velocity,
            relative_velocity_x,
            relative_velocity_y,
            distance_to_vertical_edge,
            distance_to_horizontal_edge,
            angle_between_direction_velocity
        ]

        # Get actions from neural network
        outputs = self.net.activate(state)

        # Calculate reward
        reward = 0
        done = False

        # Previous distance to target
        prev_distance = self.position.distance_to(self.target_pos)

        # Apply actions
        if outputs[0] > 0.5:
            self.angle += self.ROTATION_SPEED
        if outputs[1] > 0.5:
            self.angle -= self.ROTATION_SPEED
        self.thrust = outputs[2] > 0.5

        # Apply thrust
        if self.thrust:
            thrust_vector = pygame.math.Vector2(0, -self.THRUST).rotate(-self.angle)
            self.velocity += thrust_vector

        # Apply gravity
        self.velocity.y += self.GRAVITY

        # Limit speed
        if self.velocity.length() > self.MAX_SPEED:
            self.velocity.scale_to_length(self.MAX_SPEED)

        # Update position
        self.position += self.velocity

        # Handle wall collisions - bounce instead of teleport
        if self.position.x <= 0:
            self.position.x = 0
            self.velocity.x = abs(self.velocity.x) * 0.8  # Bounce with 80% velocity
        elif self.position.x >= self.WIDTH:
            self.position.x = self.WIDTH
            self.velocity.x = -abs(self.velocity.x) * 0.8

        if self.position.y <= 0:
            self.position.y = 0
            self.velocity.y = abs(self.velocity.y) * 0.8
        elif self.position.y >= self.HEIGHT:
            self.position.y = self.HEIGHT
            self.velocity.y = -abs(self.velocity.y) * 0.8

        # Check if rocket is nearly stationary
        if self.velocity.length() < 0.1:  # Very low speed threshold
            self.stationary_time += 1
        else:
            self.stationary_time = 0

        # Reset position if stationary for too long (5 seconds * FPS)
        if self.stationary_time > 5 * self.CLOCK_SPEED:
            reward -= 50  # Penalty for being stuck
            self.position = pygame.math.Vector2(self.WIDTH // 2, self.HEIGHT // 2)
            self.velocity = pygame.math.Vector2(0, 0)
            self.stationary_time = 0

        # Update rocket position
        self.rocket_rect.center = self.position

        # Calculate reward based on distance improvement
        current_distance = self.position.distance_to(self.target_pos)
        distance_improvement = prev_distance - current_distance
        reward += distance_improvement * 10  # Scale the reward

        # Add a small reward for proper orientation towards the moon
        angle_to_target = math.degrees(math.atan2(
            self.target_pos.y - self.position.y,
            self.target_pos.x - self.position.x
        ))
        angle_diff = abs(angle_to_target - self.angle) % 360
        if angle_diff > 180:
            angle_diff = 360 - angle_diff
        orientation_reward = (180 - angle_diff) / 180.0  # 1.0 when perfect, 0.0 when opposite
        reward += orientation_reward * 0.1

        # Add small reward for controlled movement
        speed = self.velocity.length()
        if speed < self.MAX_SPEED:
            reward += 0.1  # Small reward for controlled speed

        # Check collisions
        if self.rocket_rect.colliderect(self.moon_rect):
            reward += 100  # Big reward for reaching the moon
            self.generate_target_position()  # Generate new target

        # Update previous angle
        self.prev_angle = self.angle

        # Draw game state
        self.draw()
        pygame.display.flip()

        # Tick the clock
        self.clock.tick(self.CLOCK_SPEED)
        
        # Increment step counter
        self.steps += 1

        self.total_rewards += reward

        return state, reward, False  # Normal game step

    def draw(self):
        # Fill the screen with black color
        self.screen.fill((0, 0, 0))

        # Draw the stars
        for star in self.stars:
            x, y, size = star
            pygame.draw.circle(self.screen, (255, 255, 255), (x, y), size)

        # Draw the moon
        self.screen.blit(self.moon_img, self.moon_rect)

        # Draw the flames if thrust is applied
        if self.thrust:
            rotated_flames = pygame.transform.rotate(self.flames_img, self.angle)
            flames_offset = pygame.math.Vector2(0, self.rocket_rect.height * 0.75).rotate(-self.angle)
            flames_pos = self.rocket_rect.center + flames_offset
            flames_rect = rotated_flames.get_rect(center=flames_pos)
            self.screen.blit(rotated_flames, flames_rect)

        # Draw the rotated rocket
        rotated_rocket = pygame.transform.rotate(self.rocket_img, self.angle)
        rotated_rect = rotated_rocket.get_rect(center=self.rocket_rect.center)
        self.screen.blit(rotated_rocket, rotated_rect)

        # Display speed
        speed = self.velocity.length()
        velocity_text = f"Speed: {speed:.2f}"
        speed_surface = self.font.render(velocity_text, True, (255, 255, 255))
        self.screen.blit(speed_surface, (10, 40))

        # Render timer text
        timer_text = self.font.render(f"Time: {int(self.timer / 60):02d}.{int(self.timer % 60):02d}", True, (255, 255, 255))
        self.screen.blit(timer_text, (10, 10))

        # Display total rewards
        total_rewards_text = self.font.render(f"Total Rewards: {self.total_rewards:.2f}", True, (255, 255, 255))
        self.screen.blit(total_rewards_text, (10, 70))

        # Draw training graphs if they exist
        if hasattr(self, 'draw_graph') and hasattr(self, 'training_data'):
            # Draw smaller graphs in top right
            self.draw_graph(self.screen, self.training_data['rewards'], (600, 20), (150, 60), (0, 255, 0), "Total Rewards")
            self.draw_graph(self.screen, self.training_data['last_rewards'], (600, 100), (150, 60), (0, 0, 255), "Last Reward")

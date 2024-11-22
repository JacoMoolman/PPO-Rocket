import pygame
import math
import random

class MoonLanderGame:
    def __init__(self, network, num_landers=1):
        pygame.init()
        self.WIDTH = 800
        self.HEIGHT = 600
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption("Moon Lander")
        self.clock = pygame.time.Clock()
        self.network = network
        self.num_landers = num_landers
        self.total_score = 0
        self.steps = 0

        # Physics constants
        self.GRAVITY = 0.05  
        self.THRUST = 0.5   
        self.ROTATION_SPEED = 3
        self.MAX_SPEED = 5
        self.MAX_ANGULAR_VELOCITY = 10
        self.CLOCK_SPEED = 600

        # Load images for all landers to share
        self.rocket_img = pygame.image.load("Rocket.png")
        self.rocket_img = pygame.transform.scale(self.rocket_img, (self.rocket_img.get_width() // 2, self.rocket_img.get_height() // 2))
        
        self.flames_img = pygame.image.load("Flames.png")
        self.flames_img = pygame.transform.scale(self.flames_img, (self.rocket_img.get_width(), self.rocket_img.get_height()))
        
        self.moon_img = pygame.image.load("Moonimage.png")
        self.target_radius = 20
        self.moon_img = pygame.transform.scale(self.moon_img, (self.target_radius * 2, self.target_radius * 2))

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

        self.initialize_game()

    def initialize_game(self):
        # Initialize multiple landers
        self.landers = []
        self.lander_states = []
        
        # Generate ONE moon target
        while True:
            x = random.randint(self.target_radius, self.WIDTH - self.target_radius)
            y = random.randint(self.target_radius, self.HEIGHT - self.target_radius)
            self.target_pos = pygame.math.Vector2(x, y)
            self.moon_rect = self.moon_img.get_rect(center=self.target_pos)
            # For first moon placement, no need to check rocket positions as none exist yet
            break
        
        for _ in range(self.num_landers):
            # Create a lander with all the original graphics and properties
            lander = {
                'position': pygame.math.Vector2(random.randint(100, self.WIDTH-100), self.HEIGHT // 4),
                'velocity': pygame.math.Vector2(0, 0),
                'angle': random.uniform(-0.2, 0.2),
                'angular_velocity': 0,
                'prev_angle': 0,
                'score': 0,
                'thrust': False,
                'rocket_rect': self.rocket_img.get_rect(),
                'stationary_time': 0,
                'crashed': False,
                'landed': False
            }
            lander['rocket_rect'].center = lander['position']
            self.landers.append(lander)
            self.lander_states.append(self.get_state(lander))

    def generate_new_moon_position(self):
        # Try to find a position that's not too close to any rocket
        MAX_ATTEMPTS = 100
        MIN_DISTANCE = 100  # Minimum distance from any rocket
        
        for _ in range(MAX_ATTEMPTS):
            x = random.randint(self.target_radius, self.WIDTH - self.target_radius)
            y = random.randint(self.target_radius, self.HEIGHT - self.target_radius)
            new_pos = pygame.math.Vector2(x, y)
            
            # Check distance from all rockets
            too_close = False
            for lander in self.landers:
                if new_pos.distance_to(lander['position']) < MIN_DISTANCE:
                    too_close = True
                    break
            
            if not too_close:
                self.target_pos = new_pos
                self.moon_rect.center = self.target_pos
                return True
                
        # If we couldn't find a good spot after MAX_ATTEMPTS, just put it somewhere random
        # This should rarely happen unless the screen is very crowded
        x = random.randint(self.target_radius, self.WIDTH - self.target_radius)
        y = random.randint(self.target_radius, self.HEIGHT - self.target_radius)
        self.target_pos = pygame.math.Vector2(x, y)
        self.moon_rect.center = self.target_pos
        return True

    def get_state(self, lander):
        # Calculate inputs for neural network (same as original)
        x_distance = (self.target_pos.x - lander['position'].x) / self.WIDTH
        y_distance = (self.target_pos.y - lander['position'].y) / self.HEIGHT
        angle_to_moon = math.degrees(math.atan2(y_distance, x_distance))
        normalized_angle = angle_to_moon / 180

        elapsed_time = self.clock.get_time() / 1000
        if elapsed_time > 0:
            angular_velocity = (lander['angle'] - lander['prev_angle']) / elapsed_time
            normalized_angular_velocity = angular_velocity / self.MAX_ANGULAR_VELOCITY
        else:
            normalized_angular_velocity = 0

        relative_velocity_x = lander['velocity'].x / self.MAX_SPEED
        relative_velocity_y = lander['velocity'].y / self.MAX_SPEED
        distance_to_vertical_edge = min(lander['position'].x, self.WIDTH - lander['position'].x) / self.WIDTH
        distance_to_horizontal_edge = min(lander['position'].y, self.HEIGHT - lander['position'].y) / self.HEIGHT
        angle_between_direction_velocity = math.atan2(lander['velocity'].y, lander['velocity'].x) - math.radians(lander['angle'])

        return [
            lander['position'].x / self.WIDTH,
            lander['position'].y / self.HEIGHT,
            lander['angle'] / 360,
            lander['velocity'].x / self.MAX_SPEED,
            lander['velocity'].y / self.MAX_SPEED,
            lander['position'].distance_to(self.target_pos) / math.sqrt(self.WIDTH**2 + self.HEIGHT**2),
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

    def run_step(self):
        states = []
        rewards = []
        dones = []
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return [None] * self.num_landers, [0] * self.num_landers, [True] * self.num_landers

        # Only clear the game area (0 to 580), leave the graph area alone
        pygame.draw.rect(self.screen, (0, 0, 0), (0, 0, 580, self.HEIGHT))

        # Only draw stars in the game area
        for star in self.stars:
            x, y, size = star
            if x < 580:  # Only draw stars in the game area
                pygame.draw.circle(self.screen, (255, 255, 255), (x, y), size)

        # Update and draw each lander
        for i, lander in enumerate(self.landers):
            if lander['crashed'] or lander['landed']:
                states.append(self.get_state(lander))
                rewards.append(0)
                dones.append(True)
                continue

            # Get actions from network
            actions = self.network.activate(self.lander_states[i])
            
            # Previous distance to target
            prev_distance = lander['position'].distance_to(self.target_pos)

            # Apply actions
            if actions[0] > 0.5:
                lander['angle'] += self.ROTATION_SPEED
            if actions[1] > 0.5:
                lander['angle'] -= self.ROTATION_SPEED
            lander['thrust'] = actions[2] > 0.5

            # Apply thrust
            if lander['thrust']:
                thrust_vector = pygame.math.Vector2(0, -self.THRUST).rotate(-lander['angle'])
                lander['velocity'] += thrust_vector

            # Apply gravity and physics
            lander['velocity'].y += self.GRAVITY
            if lander['velocity'].length() > self.MAX_SPEED:
                lander['velocity'].scale_to_length(self.MAX_SPEED)
            lander['position'] += lander['velocity']

            # Boundary checks with bounce effect
            bounce_factor = -0.8  # Strong bounce but with some energy loss
            
            if lander['position'].x < 0:
                lander['position'].x = 0
                lander['velocity'].x *= bounce_factor
            elif lander['position'].x > self.WIDTH:
                lander['position'].x = self.WIDTH
                lander['velocity'].x *= bounce_factor
                
            if lander['position'].y < 0:
                lander['position'].y = 0
                lander['velocity'].y *= bounce_factor
            elif lander['position'].y > self.HEIGHT:
                lander['position'].y = self.HEIGHT
                lander['velocity'].y *= bounce_factor

            # Add a small random rotation on bounce to make it more interesting
            if (lander['position'].x <= 0 or lander['position'].x >= self.WIDTH or
                lander['position'].y <= 0 or lander['position'].y >= self.HEIGHT):
                lander['angular_velocity'] += random.uniform(-2, 2)

            # Calculate reward - just based on getting closer to target
            current_distance = lander['position'].distance_to(self.target_pos)
            distance_improvement = prev_distance - current_distance
            reward = distance_improvement * 0.1  # Small reward for getting closer
            
            # Update rocket position and check collisions
            lander['rocket_rect'].center = lander['position']
            if lander['rocket_rect'].colliderect(self.moon_rect):
                reward = 500  # Big reward for hitting the moon
                lander['score'] += reward
                self.total_score += reward
                
                # Generate new target
                self.generate_new_moon_position()

            # Draw the moon target
            self.screen.blit(self.moon_img, self.moon_rect)

            # Draw flames if thrusting
            if lander['thrust']:
                rotated_flames = pygame.transform.rotate(self.flames_img, lander['angle'])
                flames_offset = pygame.math.Vector2(0, lander['rocket_rect'].height * 0.75).rotate(-lander['angle'])
                flames_pos = lander['rocket_rect'].center + flames_offset
                flames_rect = rotated_flames.get_rect(center=flames_pos)
                self.screen.blit(rotated_flames, flames_rect)

            # Draw the rocket
            rotated_rocket = pygame.transform.rotate(self.rocket_img, lander['angle'])
            rotated_rect = rotated_rocket.get_rect(center=lander['rocket_rect'].center)
            self.screen.blit(rotated_rocket, rotated_rect)

            # Update previous angle
            lander['prev_angle'] = lander['angle']

            # Store state, reward, and done flag
            self.lander_states[i] = self.get_state(lander)
            states.append(self.lander_states[i])
            rewards.append(reward)
            dones.append(False)

        # Draw UI elements
        speed_text = f"Avg Speed: {sum(l['velocity'].length() for l in self.landers) / self.num_landers:.2f}"
        speed_surface = self.font.render(speed_text, True, (255, 255, 255))
        self.screen.blit(speed_surface, (10, 40))

        timer_text = self.font.render(f"Time: {int(self.steps / 60):02d}.{int(self.steps % 60):02d}", True, (255, 255, 255))
        self.screen.blit(timer_text, (10, 10))

        # Draw noise level if network has noise attribute
        if hasattr(self.network, 'agent') and hasattr(self.network.agent, 'noise_threshold'):
            noise_text = self.font.render(f"Noise: {self.network.agent.noise_threshold:.3f}", True, (255, 255, 255))
            self.screen.blit(noise_text, (10, 70))

        pygame.display.flip()
        self.clock.tick(self.CLOCK_SPEED)
        self.steps += 1

        return states, rewards, dones

    def reset(self):
        self.initialize_game()
        return self.lander_states

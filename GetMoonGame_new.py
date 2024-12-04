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
        self.noise_value = 0  # Add this to store noise value

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
        self.CLOCK_SPEED = 60  # Reduced from 400 to 60 for smoother display

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
        self.GRAVITY = 0.05  
        self.THRUST = 0.5
        self.ROTATION_SPEED = 3
        self.MAX_SPEED = 3
        self.MAX_ANGULAR_VELOCITY = 10

        # Initialize font
        pygame.font.init()
        self.font = pygame.font.Font(None, 36)

        # Initialize rocket trails
        self.trail_points = []  # List to store trail points and their opacity
        self.MAX_TRAIL_LENGTH = 20  # Maximum number of trail points to store

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
        
        # Reset steps counter
        self.steps = 0

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
        # 1. Position relative to target (normalized)
        x_distance = (self.target_pos.x - self.position.x) / self.WIDTH
        y_distance = (self.target_pos.y - self.position.y) / self.HEIGHT
        
        # 2. Relative angle to target (normalized to [-1, 1])
        angle_to_moon = math.degrees(math.atan2(y_distance, x_distance))
        relative_angle = (angle_to_moon - self.angle) % 360
        if relative_angle > 180:
            relative_angle = relative_angle - 360
        normalized_angle = relative_angle / 180
        
        # 3. Current velocity (normalized)
        normalized_velocity_x = self.velocity.x / self.MAX_SPEED
        normalized_velocity_y = self.velocity.y / self.MAX_SPEED
        
        # 4. Angular velocity (normalized)
        elapsed_time = self.clock.get_time() / 1000
        if elapsed_time > 0:
            angular_velocity = (self.angle - self.prev_angle) / elapsed_time
            normalized_angular_velocity = angular_velocity / self.MAX_ANGULAR_VELOCITY
        else:
            normalized_angular_velocity = 0
            
        # 5. Distance to walls (normalized)
        distance_to_vertical_edge = min(self.position.x, self.WIDTH - self.position.x) / self.WIDTH
        distance_to_horizontal_edge = min(self.position.y, self.HEIGHT - self.position.y) / self.HEIGHT

        state = [
            x_distance,                    # Distance to target X
            y_distance,                    # Distance to target Y
            normalized_angle,              # Angle needed to face target
            normalized_velocity_x,         # Current X velocity
            normalized_velocity_y,         # Current Y velocity
            normalized_angular_velocity,   # Current rotation speed
            distance_to_vertical_edge,     # Distance to left/right walls
            distance_to_horizontal_edge,   # Distance to top/bottom walls
        ]

        # Get actions from neural network
        outputs = self.net.activate(state)

        # Calculate reward
        reward = 0
        done = False

        # Previous distance to target
        prev_distance = self.position.distance_to(self.target_pos)

        # Apply actions
        if outputs[0] or outputs[1]:  # Strong or weak right rotation
            self.angle += self.ROTATION_SPEED
        if outputs[2] or outputs[3]:  # Strong or weak left rotation
            self.angle -= self.ROTATION_SPEED
        self.angle = self.angle % 360  # Normalize angle to 0-360 range
        self.thrust = outputs[4] or outputs[5] or outputs[6]  # Any level of thrust

        # Apply thrust
        if self.thrust:
            thrust_vector = pygame.math.Vector2(0, -self.THRUST).rotate(-self.angle)  # Put back the negative sign
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
            reward -= 100  # Much bigger penalty for being stuck
            return state, reward, True  # End episode when stuck

        # Update rocket position
        self.rocket_rect.center = self.position

        # Update trail points
        if self.thrust:
            # Calculate the position for the trail (at the bottom of the rocket)
            trail_offset = pygame.math.Vector2(0, self.rocket_rect.height * 0.5).rotate(-self.angle)
            trail_pos = self.position + trail_offset
            
            # Add new trail point with full opacity
            self.trail_points.append({"pos": trail_pos, "opacity": 255})
            
            # Remove old trail points if we exceed the maximum length
            if len(self.trail_points) > self.MAX_TRAIL_LENGTH:
                self.trail_points.pop(0)
        
        # Fade out existing trail points
        for point in self.trail_points:
            point["opacity"] = max(0, point["opacity"] - 10)  # Decrease opacity
        
        # Remove completely faded points
        self.trail_points = [p for p in self.trail_points if p["opacity"] > 0]

        # Calculate reward based on distance improvement
        current_distance = self.position.distance_to(self.target_pos)
        distance_improvement = prev_distance - current_distance
        reward += distance_improvement * 5  # Keep the distance improvement reward

        # Penalty for moving away from moon
        if distance_improvement < 0:
            reward += distance_improvement * 10  # Additional penalty for moving away

        # Calculate angle to moon for rewards
        dx = self.target_pos.x - self.position.x
        dy = self.target_pos.y - self.position.y
        angle_to_moon = math.degrees(math.atan2(-dy, dx))  # Negative dy because pygame y increases downward
        relative_angle = (self.angle - angle_to_moon + 90) % 360  # +90 because rocket points up at 0
        if relative_angle > 180:
            relative_angle = 360 - relative_angle
            
        # Simple reward for pointing at moon: +10 when perfect, -10 when opposite
        reward += (180 - relative_angle) / 18.0  # Scales from -10 to +10

        # Penalty for thrusting in wrong direction
        if self.thrust and relative_angle > 90:
            reward -= 5  # Big penalty for thrusting while pointing away

        # Time-based penalties
        reward -= 0.1  # Small constant penalty per timestep
        
        # Path efficiency penalty
        straight_line_distance = math.sqrt(dx * dx + dy * dy)
        if self.steps > 0:
            efficiency_penalty = -0.01 * (self.steps - straight_line_distance / self.MAX_SPEED)
            reward += max(efficiency_penalty, -1)  # Cap the penalty at -1 per step

        # Check collisions with better reward structure
        if self.rocket_rect.colliderect(self.moon_rect):
            # Base reward for hitting the target
            reward += 1000
            
            # Time bonus - more reward for faster completion
            time_bonus = max(0, 2000 - (self.steps * 2))  # Starts at 2000, decreases by 2 per step
            reward += time_bonus
            
            print(f"HIT MOON - Steps: {self.steps}, Time Bonus: {time_bonus:.2f}")
            # Generate new target position and reset steps counter
            self.generate_target_position()
            self.steps = 0  # Reset steps counter when moon is hit

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

        # Draw the trails
        for point in self.trail_points:
            pygame.draw.circle(self.screen, (255, 165, 0, point["opacity"]), (int(point["pos"].x), int(point["pos"].y)), 2)

        # Draw the moon
        self.screen.blit(self.moon_img, self.moon_rect)

        # Draw the rocket
        rotated_rocket = pygame.transform.rotate(self.rocket_img, self.angle)
        rotated_rect = rotated_rocket.get_rect(center=self.rocket_rect.center)
        self.screen.blit(rotated_rocket, rotated_rect)

        # Draw flames if thrusting
        if self.thrust:
            rotated_flames = pygame.transform.rotate(self.flames_img, self.angle)
            flames_rect = rotated_flames.get_rect(center=self.rocket_rect.center)
            flames_rect.centerx += math.sin(math.radians(self.angle)) * self.rocket_rect.height / 2
            flames_rect.centery += math.cos(math.radians(self.angle)) * self.rocket_rect.height / 2
            self.screen.blit(rotated_flames, flames_rect)

        # Draw text information
        font = pygame.font.Font(None, 36)
        
        # Display total rewards
        total_rewards_text = self.font.render(f"Total Rewards: {self.total_rewards:.2f}", True, (255, 255, 255))
        self.screen.blit(total_rewards_text, (10, 10))
        
        # Display time counter
        time_text = self.font.render(f"Time: {self.steps}", True, (255, 255, 255))
        self.screen.blit(time_text, (10, 50))
        
        # Display noise value
        noise_text = self.font.render(f"Noise: {self.noise_value:.5f}", True, (255, 255, 255))
        self.screen.blit(noise_text, (10, 90))
        
        # Display angle
        angle_text = self.font.render(f"Angle: {self.angle:.2f}", True, (255, 255, 255))
        self.screen.blit(angle_text, (10, 130))
        
        # Display velocity/speed
        speed = self.velocity.length()
        speed_text = self.font.render(f"Speed: {speed:.2f}", True, (255, 255, 255))
        self.screen.blit(speed_text, (10, 170))

        # Display angle to moon
        dx = self.target_pos.x - self.position.x
        dy = self.target_pos.y - self.position.y
        angle_to_moon = math.degrees(math.atan2(-dy, dx))  # Negative dy because pygame y increases downward
        relative_angle = (self.angle - angle_to_moon + 90) % 360  # +90 because rocket points up at 0
        if relative_angle > 180:
            relative_angle = 360 - relative_angle
        moon_angle_text = self.font.render(f"Moon Angle: {relative_angle:.2f}", True, (255, 255, 255))
        self.screen.blit(moon_angle_text, (10, 210))

        # Draw training graphs if they exist
        if hasattr(self, 'draw_graph') and hasattr(self, 'training_data'):
            # Draw smaller graphs in top right
            self.draw_graph(self.screen, self.training_data['rewards'], (600, 20), (150, 60), (0, 255, 0), "Total Rewards")
            self.draw_graph(self.screen, self.training_data['last_rewards'], (600, 100), (150, 60), (0, 0, 255), "Last Reward")

    def update_noise(self, noise):
        self.noise_value = noise

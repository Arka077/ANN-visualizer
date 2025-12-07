"""
Smooth Neural Network Visualizer - Pure Python with Pygame
Dataset: Wine Classification (13 → 6 → 4 → 3)

Features:
- Smooth, interpolated value transitions (no jumps!)
- Animated curves (loss and accuracy)
- Gradual color changes on nodes
- Smooth weight updates on edges
- Particle effects for data flow
- Perfect for demonstrations and teaching

Run: python nn_visualizer_smooth.py
"""

import pygame
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
from collections import deque
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ==================== CONFIG ====================
WIDTH, HEIGHT = 1280, 720
FPS = 60
BG_COLOR = (26, 26, 46)

# Network architecture
LAYERS = [13, 6, 4, 3]

# Training
LEARNING_RATE = 0.001
BATCH_SIZE = 2
EPOCHS = 80

# Animation
LERP_SPEED = 4.0  # Higher = faster transitions
NODE_RADIUS = 25  # Smaller nodes for compact display
PARTICLE_SPEED = 200.0

# ==================== ANIMATED VALUE CLASS ====================
class AnimatedValue:
    """Smoothly interpolates between values"""
    def __init__(self, initial=0.0):
        self.current = float(initial)
        self.target = float(initial)
    
    def set_target(self, new_target):
        self.target = float(new_target)
    
    def update(self, dt):
        diff = self.target - self.current
        self.current += diff * LERP_SPEED * dt
        
        # Snap when very close
        if abs(diff) < 0.001:
            self.current = self.target
    
    def get(self):
        return self.current

# ==================== COLOR INTERPOLATION ====================
def lerp_color(color1, color2, t):
    """Blend between two RGB colors"""
    t = max(0, min(1, t))
    r = int(color1[0] + (color2[0] - color1[0]) * t)
    g = int(color1[1] + (color2[1] - color1[1]) * t)
    b = int(color1[2] + (color2[2] - color1[2]) * t)
    return (r, g, b)

def get_activation_color(value):
    """Get color based on activation value"""
    # Positive activations: dark blue -> bright cyan
    # Negative activations: dark red -> bright red
    if value > 0:
        t = min(1.0, value / 2.0)
        dark_blue = (20, 50, 100)
        bright_cyan = (0, 217, 255)
        return lerp_color(dark_blue, bright_cyan, t)
    else:
        t = min(1.0, abs(value) / 2.0)
        dark_red = (100, 20, 20)
        bright_red = (255, 107, 107)
        return lerp_color(dark_red, bright_red, t)

def get_weight_color(value):
    """Get color based on weight value"""
    if value > 0:
        return (0, 255, 136)  # Green
    else:
        return (255, 107, 107)  # Red

# ==================== NEURAL NETWORK MODEL ====================
class WineANN(nn.Module):
    def __init__(self, layer_sizes):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
        self.activations = [None] * len(layer_sizes)
        self.layer_sizes = layer_sizes
    
    def forward(self, x):
        self.activations[0] = x[0].detach().cpu().numpy() if len(x.shape) > 1 else x.detach().cpu().numpy()
        out = x
        for i, layer in enumerate(self.layers, start=1):
            out = layer(out)
            if i < len(self.layers):
                out = torch.relu(out)
            self.activations[i] = out[0].detach().cpu().numpy() if len(out.shape) > 1 else out.detach().cpu().numpy()
        return out

# ==================== PARTICLE SYSTEM ====================
class Particle:
    def __init__(self, start_pos, end_pos, color):
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.color = color
        self.progress = 0.0
        self.speed = PARTICLE_SPEED
        self.alive = True
    
    def update(self, dt):
        distance = math.sqrt((self.end_pos[0] - self.start_pos[0])**2 + 
                           (self.end_pos[1] - self.start_pos[1])**2)
        self.progress += (self.speed / max(distance, 1)) * dt
        
        if self.progress >= 1.0:
            self.alive = False
    
    def draw(self, screen):
        if not self.alive:
            return
        
        t = self.progress
        x = self.start_pos[0] + (self.end_pos[0] - self.start_pos[0]) * t
        y = self.start_pos[1] + (self.end_pos[1] - self.start_pos[1]) * t
        
        # Draw glowing particle
        for radius in [8, 6, 4]:
            alpha = 255 - (8 - radius) * 30
            color = (*self.color, alpha)
            s = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
            pygame.draw.circle(s, color, (radius, radius), radius)
            screen.blit(s, (int(x - radius), int(y - radius)))

# ==================== VISUALIZER ====================
class SmoothVisualizer:
    def __init__(self, layer_sizes):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Neural Network - Smooth Animation")
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.SysFont('Arial', 16, bold=True)
        self.font_medium = pygame.font.SysFont('Arial', 13)
        self.font_small = pygame.font.SysFont('Arial', 11)
        
        self.layer_sizes = layer_sizes
        self.node_positions = self.compute_positions()
        
        # Animated values for smooth transitions
        self.animated_activations = []
        for size in layer_sizes:
            self.animated_activations.append([AnimatedValue() for _ in range(size)])
        
        self.animated_weights = []
        for i in range(len(layer_sizes) - 1):
            layer_weights = []
            for j in range(layer_sizes[i+1]):
                layer_weights.append([AnimatedValue() for _ in range(layer_sizes[i])])
            self.animated_weights.append(layer_weights)
        
        # Metrics
        self.loss_history = deque(maxlen=200)
        self.train_acc_history = deque(maxlen=200)
        self.test_acc_history = deque(maxlen=200)
        self.animated_loss = AnimatedValue()
        self.animated_train_acc = AnimatedValue()
        self.animated_test_acc = AnimatedValue()
        
        # Particles
        self.particles = []
        
        # UI state
        self.running = True
        self.show_weights = True
        self.show_particles = True
        
        # Buttons
        self.buttons = self.create_buttons()
    
    def compute_positions(self):
        """Compute node positions for visualization"""
        positions = []
        network_left = 80
        network_width = 620  # Reduced to fit screen
        network_top = 120
        network_height = 520  # Reduced to fit screen
        
        n_layers = len(self.layer_sizes)
        
        for i, size in enumerate(self.layer_sizes):
            x = network_left + (i / (n_layers - 1)) * network_width
            layer_positions = []
            
            for j in range(size):
                y = network_top + (j + 1) / (size + 1) * network_height
                layer_positions.append((int(x), int(y)))
            
            positions.append(layer_positions)
        
        return positions
    
    def create_buttons(self):
        buttons = {}
        x_start = WIDTH - 240
        y_start = 40
        button_height = 35
        gap = 12
        
        labels = ["Start/Pause", "Show Weights", "Show Particles"]
        for i, label in enumerate(labels):
            rect = pygame.Rect(x_start, y_start + i * (button_height + gap), 220, button_height)
            buttons[label] = rect
        
        return buttons
    
    def update_values(self, weights, activations):
        """Set target values for smooth animation"""
        # Update activations
        for layer_idx, layer_acts in enumerate(activations):
            if layer_acts is not None:
                for node_idx, value in enumerate(layer_acts):
                    if node_idx < len(self.animated_activations[layer_idx]):
                        self.animated_activations[layer_idx][node_idx].set_target(value)
        
        # Update weights
        for layer_idx, layer_weights in enumerate(weights):
            for out_idx in range(layer_weights.shape[0]):
                for in_idx in range(layer_weights.shape[1]):
                    if (out_idx < len(self.animated_weights[layer_idx]) and 
                        in_idx < len(self.animated_weights[layer_idx][out_idx])):
                        self.animated_weights[layer_idx][out_idx][in_idx].set_target(
                            layer_weights[out_idx, in_idx]
                        )
    
    def update_metrics(self, loss, train_acc, test_acc):
        """Update metric values for smooth animation"""
        self.animated_loss.set_target(loss)
        self.animated_train_acc.set_target(train_acc)
        self.animated_test_acc.set_target(test_acc)
        
        self.loss_history.append(loss)
        self.train_acc_history.append(train_acc)
        self.test_acc_history.append(test_acc)
    
    def spawn_particles(self):
        """Create particles for forward pass animation"""
        if not self.show_particles:
            return
        
        # Spawn particles from ALL nodes in first layer to random nodes in second layer
        for i in range(len(self.node_positions[0])):  # All input nodes
            # Pick 1-2 random target nodes
            num_targets = np.random.randint(1, 3)
            target_indices = np.random.choice(len(self.node_positions[1]), num_targets, replace=False)
            
            for j in target_indices:
                start = self.node_positions[0][i]
                end = self.node_positions[1][j]
                color = (255, 220, 100)
                self.particles.append(Particle(start, end, color))
    
    def update(self, dt):
        """Update all animated values"""
        # Update activations
        for layer in self.animated_activations:
            for anim_val in layer:
                anim_val.update(dt)
        
        # Update weights
        for layer in self.animated_weights:
            for row in layer:
                for anim_val in row:
                    anim_val.update(dt)
        
        # Update metrics
        self.animated_loss.update(dt)
        self.animated_train_acc.update(dt)
        self.animated_test_acc.update(dt)
        
        # Update particles
        self.particles = [p for p in self.particles if p.alive]
        for particle in self.particles:
            particle.update(dt)
    
    def draw(self):
        """Draw everything"""
        self.screen.fill(BG_COLOR)
        
        # Draw title
        title = self.font_large.render("Wine Classification Neural Network", True, (0, 217, 255))
        self.screen.blit(title, (30, 20))
        
        subtitle = self.font_medium.render("13 → 6 → 4 → 3 (Smooth Animation)", True, (140, 140, 140))
        self.screen.blit(subtitle, (30, 45))
        
        # Draw network
        self.draw_edges()
        self.draw_nodes()
        
        # Draw particles
        for particle in self.particles:
            particle.draw(self.screen)
        
        # Draw layer labels
        self.draw_layer_labels()
        
        # Draw metrics
        self.draw_metrics()
        
        # Draw buttons
        self.draw_buttons()
        
        pygame.display.flip()
    
    def draw_edges(self):
        """Draw edges with smooth weight visualization"""
        for layer_idx in range(len(self.node_positions) - 1):
            src_positions = self.node_positions[layer_idx]
            dst_positions = self.node_positions[layer_idx + 1]
            
            for out_idx, (x2, y2) in enumerate(dst_positions):
                for in_idx, (x1, y1) in enumerate(src_positions):
                    # Get smoothly animated weight
                    weight_val = self.animated_weights[layer_idx][out_idx][in_idx].get()
                    
                    # Skip very small weights
                    if abs(weight_val) < 0.2:
                        continue
                    
                    # Color and width based on weight
                    color = get_weight_color(weight_val)
                    alpha = min(200, int(abs(weight_val) * 100))
                    width = max(1, int(abs(weight_val) * 4))
                    
                    # Draw edge with alpha
                    surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
                    pygame.draw.line(surf, (*color, alpha), (x1, y1), (x2, y2), width)
                    self.screen.blit(surf, (0, 0))
                    
                    # Draw weight value if enabled and significant
                    if self.show_weights and abs(weight_val) > 0.5:
                        mid_x = (x1 + x2) // 2
                        mid_y = (y1 + y2) // 2
                        text = self.font_small.render(f"{weight_val:.2f}", True, (200, 200, 200))
                        
                        # Background for text
                        text_rect = text.get_rect(center=(mid_x, mid_y))
                        pygame.draw.rect(self.screen, BG_COLOR, text_rect.inflate(6, 4))
                        pygame.draw.rect(self.screen, color, text_rect.inflate(6, 4), 1)
                        self.screen.blit(text, text_rect)
    
    def draw_nodes(self):
        """Draw nodes with smooth color transitions"""
        for layer_idx, layer_positions in enumerate(self.node_positions):
            for node_idx, (x, y) in enumerate(layer_positions):
                # Get smoothly animated activation
                act_val = self.animated_activations[layer_idx][node_idx].get()
                
                # Get smooth color
                color = get_activation_color(act_val)
                
                # Draw node with glow effect
                for radius in [NODE_RADIUS + 6, NODE_RADIUS + 3, NODE_RADIUS]:
                    alpha = 30 if radius > NODE_RADIUS else 255
                    s = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
                    pygame.draw.circle(s, (*color, alpha), (radius, radius), radius)
                    self.screen.blit(s, (x - radius, y - radius))
                
                # Draw border
                pygame.draw.circle(self.screen, (0, 217, 255), (x, y), NODE_RADIUS, 3)
                
                # Draw activation value
                text = self.font_medium.render(f"{act_val:.2f}", True, (255, 255, 255))
                text_rect = text.get_rect(center=(x, y))
                self.screen.blit(text, text_rect)
    
    def draw_layer_labels(self):
        """Draw labels for each layer"""
        labels = [
            f"Input\n({self.layer_sizes[0]})",
            f"Hidden 1\n({self.layer_sizes[1]})",
            f"Hidden 2\n({self.layer_sizes[2]})",
            f"Output\n({self.layer_sizes[3]})"
        ]
        
        for layer_idx, label in enumerate(labels):
            if self.node_positions[layer_idx]:
                x = self.node_positions[layer_idx][0][0]
                y = 90
                
                # Draw background
                lines = label.split('\n')
                for i, line in enumerate(lines):
                    text = self.font_small.render(line, True, (0, 217, 255))
                    text_rect = text.get_rect(center=(x, y + i * 16))
                    pygame.draw.rect(self.screen, (30, 40, 60), text_rect.inflate(8, 4))
                    pygame.draw.rect(self.screen, (0, 217, 255), text_rect.inflate(8, 4), 1)
                    self.screen.blit(text, text_rect)
    
    def draw_metrics(self):
        """Draw loss and accuracy curves"""
        # Define chart areas - adjusted for 1280x720
        chart_x = 750
        chart_width = 500
        
        loss_y = 140
        acc_y = 410
        chart_height = 240
        
        # Draw loss chart
        self.draw_chart(
            chart_x, loss_y, chart_width, chart_height,
            "Training Loss",
            self.loss_history,
            (255, 107, 107),
            self.animated_loss.get()
        )
        
        # Draw accuracy chart
        self.draw_chart(
            chart_x, acc_y, chart_width, chart_height,
            "Accuracy",
            self.train_acc_history,
            (0, 255, 136),
            self.animated_train_acc.get(),
            self.test_acc_history,
            (0, 217, 255),
            self.animated_test_acc.get()
        )
    
    def draw_chart(self, x, y, width, height, title, data1, color1, current1, 
                   data2=None, color2=None, current2=None):
        """Draw a chart with smooth curves"""
        # Background
        pygame.draw.rect(self.screen, (30, 40, 60), (x, y, width, height))
        pygame.draw.rect(self.screen, (80, 90, 110), (x, y, width, height), 2)
        
        # Title
        text = self.font_medium.render(title, True, (0, 217, 255))
        self.screen.blit(text, (x + 10, y + 8))
        
        # Current value
        if "Loss" in title:
            value_text = self.font_small.render(f"Current: {current1:.4f}", True, (200, 200, 200))
        else:
            value_text = self.font_small.render(
                f"Train: {current1:.1f}% | Test: {current2:.1f}%" if current2 else f"{current1:.1f}%",
                True, (200, 200, 200)
            )
        self.screen.blit(value_text, (x + 10, y + 28))
        
        # Draw grid
        for i in range(5):
            grid_y = y + 55 + (height - 75) * i // 4
            pygame.draw.line(self.screen, (60, 70, 90), (x + 15, grid_y), (x + width - 15, grid_y), 1)
        
        # Draw data with fixed scales
        if len(data1) > 1:
            if "Loss" in title:
                # Loss: zoom out to 0-2.0 range
                points1 = self.get_chart_points(data1, x + 15, y + 55, width - 30, height - 75, min_val=0, max_val=2.0)
            else:
                # Accuracy: fixed 0-100% range
                points1 = self.get_chart_points(data1, x + 15, y + 55, width - 30, height - 75, min_val=0, max_val=100)
            
            if len(points1) > 1:
                pygame.draw.lines(self.screen, color1, False, points1, 2)
        
        if data2 and len(data2) > 1:
            # Accuracy: fixed 0-100% range
            points2 = self.get_chart_points(data2, x + 15, y + 55, width - 30, height - 75, min_val=0, max_val=100)
            if len(points2) > 1:
                pygame.draw.lines(self.screen, color2, False, points2, 2)
        
        # Legend for accuracy
        if data2:
            legend_y = y + height - 20
            pygame.draw.line(self.screen, color1, (x + 15, legend_y), (x + 40, legend_y), 2)
            text = self.font_small.render("Train", True, (200, 200, 200))
            self.screen.blit(text, (x + 45, legend_y - 7))
            
            pygame.draw.line(self.screen, color2, (x + 95, legend_y), (x + 120, legend_y), 2)
            text = self.font_small.render("Test", True, (200, 200, 200))
            self.screen.blit(text, (x + 125, legend_y - 7))
    
    def get_chart_points(self, data, x, y, width, height, min_val=None, max_val=None):
        """Convert data to screen coordinates with optional fixed scales"""
        if not data:
            return []
        
        data_list = list(data)
        
        # Use provided ranges or auto-scale
        if min_val is None or max_val is None:
            min_val = min(data_list) if min_val is None else min_val
            max_val = max(data_list) if max_val is None else max_val
        
        range_val = max_val - min_val if max_val > min_val else 1.0
        
        points = []
        for i, value in enumerate(data_list):
            px = x + (i / max(len(data_list) - 1, 1)) * width
            py = y + height - ((value - min_val) / range_val) * height
            points.append((int(px), int(py)))
        
        return points
    
    def draw_buttons(self):
        """Draw UI buttons"""
        for label, rect in self.buttons.items():
            # Determine button state
            if label == "Start/Pause":
                active = self.running
            elif label == "Show Weights":
                active = self.show_weights
            elif label == "Show Particles":
                active = self.show_particles
            else:
                active = False
            
            # Draw button
            color = (60, 140, 60) if active else (60, 60, 80)
            pygame.draw.rect(self.screen, color, rect)
            pygame.draw.rect(self.screen, (120, 120, 140), rect, 2)
            
            # Draw text
            text = self.font_medium.render(label, True, (255, 255, 255))
            text_rect = text.get_rect(center=rect.center)
            self.screen.blit(text, text_rect)
    
    def handle_click(self, pos):
        """Handle button clicks"""
        for label, rect in self.buttons.items():
            if rect.collidepoint(pos):
                if label == "Start/Pause":
                    self.running = not self.running
                elif label == "Show Weights":
                    self.show_weights = not self.show_weights
                elif label == "Show Particles":
                    self.show_particles = not self.show_particles
                return True
        return False

# ==================== TRAINER ====================
class Trainer:
    def __init__(self, layer_sizes):
        self.device = torch.device('cpu')  # Use CPU for stability
        self.model = WineANN(layer_sizes).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.opt = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        
        print("=" * 70)
        print("🍷 Wine Classification Neural Network - Smooth Animation")
        print("=" * 70)
        
        self.load_dataset()
        self.vis = SmoothVisualizer(layer_sizes)
        
        self.batch_counter = 0
        self.particle_spawn_counter = 0
    
    def load_dataset(self):
        wine = load_wine()
        X = wine.data.astype(np.float32)
        y = wine.target.astype(np.int64)
        
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )
        
        print(f"Dataset: {len(self.X_train)} train, {len(self.X_test)} test")
        print(f"Features: {X.shape[1]}, Classes: 3")
        print("=" * 70)
        print("Controls:")
        print("  - Click 'Start/Pause' to begin training")
        print("  - Toggle weight display and particles")
        print("  - Close window to exit")
        print("=" * 70)
    
    def calculate_accuracy(self, X, y):
        correct = 0
        with torch.no_grad():
            for i in range(len(X)):
                x = torch.tensor(X[i], dtype=torch.float32).unsqueeze(0).to(self.device)
                out = self.model(x)
                pred = torch.argmax(out, dim=1).item()
                if pred == y[i]:
                    correct += 1
        return (correct / len(X)) * 100.0
    
    def train_step(self):
        """Perform one training batch"""
        # Get random batch
        indices = np.random.choice(len(self.X_train), BATCH_SIZE, replace=False)
        batch_x = torch.tensor(self.X_train[indices], dtype=torch.float32).to(self.device)
        batch_y = torch.tensor(self.y_train[indices], dtype=torch.long).to(self.device)
        
        # Forward and backward
        self.opt.zero_grad()
        out = self.model(batch_x)
        loss = self.criterion(out, batch_y)
        loss.backward()
        self.opt.step()
        
        # Get visualization data
        weights = [layer.weight.detach().cpu().numpy() for layer in self.model.layers]
        
        # Get activations from first sample
        sample_x = torch.tensor(self.X_train[indices[0]], dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            _ = self.model(sample_x)
        activations = self.model.activations
        
        # Calculate accuracies
        train_acc = self.calculate_accuracy(self.X_train[:50], self.y_train[:50])  # Sample for speed
        test_acc = self.calculate_accuracy(self.X_test, self.y_test)
        
        return loss.item(), train_acc, test_acc, weights, activations
    
    def run(self):
        """Main training loop with Pygame"""
        running = True
        last_time = pygame.time.get_ticks() / 1000.0
        
        # Initialize metrics
        loss = 0.0
        train_acc = 0.0
        test_acc = 0.0
        epoch = 0
        training_complete = False
        
        while running:
            current_time = pygame.time.get_ticks() / 1000.0
            dt = current_time - last_time
            last_time = current_time
            
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self.vis.handle_click(event.pos)
            
            # Training step if running
            if self.vis.running and not training_complete:
                loss, train_acc, test_acc, weights, activations = self.train_step()
                
                # Update visualizer with smooth transitions
                self.vis.update_values(weights, activations)
                self.vis.update_metrics(loss, train_acc, test_acc)
                
                # Spawn particles occasionally
                self.particle_spawn_counter += 1
                if self.particle_spawn_counter >= 10:
                    self.vis.spawn_particles()
                    self.particle_spawn_counter = 0
                
                self.batch_counter += 1
                
                # Move to next epoch after training samples
                if self.batch_counter % 10 == 0:
                    epoch += 1
                    if epoch >= EPOCHS:
                        training_complete = True
                        self.vis.running = False
                        print(f"\nTraining complete! {EPOCHS} epochs finished.")
                
                if self.batch_counter % 50 == 0:
                    print(f"Epoch {epoch}/{EPOCHS} - Batch {self.batch_counter} - Loss: {loss:.4f}, "
                          f"Train: {train_acc:.1f}%, Test: {test_acc:.1f}%")
            
            # Update animations
            self.vis.update(dt)
            
            # Draw
            self.vis.draw()
            self.vis.clock.tick(FPS)
        
        pygame.quit()
        
        print("\n" + "=" * 70)
        print("Training stopped!")
        print(f"Epochs completed: {epoch}/{EPOCHS}")
        print(f"Final Train Accuracy: {train_acc:.2f}%")
        print(f"Final Test Accuracy: {test_acc:.2f}%")
        print("=" * 70)

# ==================== MAIN ====================
def main():
    trainer = Trainer(LAYERS)
    trainer.run()

if __name__ == '__main__':
    main()
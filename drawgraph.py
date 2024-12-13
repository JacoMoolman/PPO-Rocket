import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, ArrowStyle
from matplotlib.lines import Line2D

# ---------------------------
# Helper Functions
# ---------------------------

def draw_box(ax, center, width, height, label, facecolor='#ffffff'):
    """Draw a rounded box with a label."""
    x, y = center
    rect = FancyBboxPatch((x - width/2, y - height/2), width, height,
                          boxstyle="round,pad=0.1", 
                          linewidth=1, edgecolor='black', facecolor=facecolor)
    ax.add_patch(rect)
    ax.text(x, y, label, ha='center', va='center', fontsize=10)

def draw_arrow(ax, start, end):
    """Draw an arrow from start=(x1, y1) to end=(x2, y2)."""
    style = ArrowStyle("->", head_width=0.2, head_length=0.4)
    ax.annotate("", xy=end, xytext=start,
                arrowprops=dict(arrowstyle=style, color='black', linewidth=1))

# ---------------------------
# Figure Setup
# ---------------------------
fig, ax = plt.subplots(figsize=(12, 6))
ax.set_xlim(-1, 21)
ax.set_ylim(-3, 3)
ax.axis('off')

# We have:
# - Input Layer (1)
# - Policy Network: 10 Linear+Tanh pairs (represented as blocks)
# - Action Head
# - Value Network: 10 Linear+Tanh pairs
# - Value Head
#
# We'll represent each linear+tanh pair as a single box for brevity.
# Input at x=0, each subsequent layer step at +2 x units.

# Positions
x_input = 0
x_layers = [x_input + 2*(i+1) for i in range(10)]  # 10 intermediate steps
x_output_policy = x_layers[-1] + 2
x_output_value = x_layers[-1] + 2

y_input = 0.0
y_policy = 1.5
y_value = -1.5

box_width = 1.4
box_height = 0.8

# Colors
input_color = '#cce5ff'
policy_color = '#ffd2cc'
value_color = '#ccffcc'
output_color = '#ffffcc'

# ---------------------------
# Draw Input
# ---------------------------
draw_box(ax, (x_input, y_input), box_width, box_height, "Input\n(8-dim)", facecolor=input_color)

# ---------------------------
# Draw Policy Layers
# ---------------------------
for i, x_pos in enumerate(x_layers):
    draw_box(ax, (x_pos, y_policy), box_width, box_height, 
             f"Policy L{i+1}\n(Linear+Tanh 512)", facecolor=policy_color)

# Policy Action Head
draw_box(ax, (x_output_policy, y_policy), box_width, box_height, 
         "Action Head\n(Linear 512->3)", facecolor=output_color)

# ---------------------------
# Draw Value Layers
# ---------------------------
for i, x_pos in enumerate(x_layers):
    draw_box(ax, (x_pos, y_value), box_width, box_height, 
             f"Value L{i+1}\n(Linear+Tanh 512)", facecolor=value_color)

# Value Head
draw_box(ax, (x_output_value, y_value), box_width, box_height, 
         "Value Head\n(Linear 512->1)", facecolor=output_color)

# ---------------------------
# Draw Arrows
# ---------------------------
# From input to first policy and value layer
draw_arrow(ax, (x_input + box_width/2, y_input), (x_layers[0] - box_width/2, y_policy))
draw_arrow(ax, (x_input + box_width/2, y_input), (x_layers[0] - box_width/2, y_value))

# Chain policy layers
for i in range(len(x_layers)-1):
    draw_arrow(ax, (x_layers[i] + box_width/2, y_policy), (x_layers[i+1] - box_width/2, y_policy))
# From last policy layer to Action Head
draw_arrow(ax, (x_layers[-1] + box_width/2, y_policy), (x_output_policy - box_width/2, y_policy))

# Chain value layers
for i in range(len(x_layers)-1):
    draw_arrow(ax, (x_layers[i] + box_width/2, y_value), (x_layers[i+1] - box_width/2, y_value))
# From last value layer to Value Head
draw_arrow(ax, (x_layers[-1] + box_width/2, y_value), (x_output_value - box_width/2, y_value))

ax.set_title("Policy-Value Network Architecture", fontsize=14, pad=20)
plt.tight_layout()
plt.show()

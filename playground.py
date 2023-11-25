import matplotlib.pyplot as plt
import numpy as np

# Sample data
x = np.linspace(1, 10, 100)
y = np.power(3, x)  # Exponential growth for demonstration

plt.figure(figsize=(10, 6))
plt.plot(x, y)

# Set y-axis to log scale (default base 10)
plt.yscale('log')

# Get current axis
ax = plt.gca()

# Get current y-ticks (which are in base 10)
y_vals = ax.get_yticks()

# Convert y-tick labels to base 3 log
ax.set_yticklabels([f"{np.log(val)/np.log(3):.2f}" if val > 0 else '0' for val in y_vals])

plt.xlabel('X-axis')
plt.ylabel('log3(Y-axis)')
plt.title('Y-axis in log3 Scale')
plt.grid(True)
plt.show()

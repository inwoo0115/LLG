import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def llg_simulation(magnetization, h_eff, damping, gyromagnetic_ratio, time_step, total_time):
    # magnetization = magnetization / np.linalg.norm(magnetization)  # Normalize magnetization vector

    num_steps = int(total_time / time_step)
    magnetization_history = np.zeros((num_steps + 1, 3))
    magnetization_history[0] = magnetization

    for i in range(num_steps):
        m = magnetization
        magnetization += time_step * np.cross(m, h_eff) * (-1) * gyromagnetic_ratio
        magnetization -= np.cross(damping * m, np.cross(m, h_eff)) # Damping parameter
        # magnetization /= np.linalg.norm(magnetization)  # Normalize magnetization vector
        magnetization_history[i + 1] = magnetization

    return magnetization_history

def plot_magnetization(magnetization_history, time_step):
    time = np.arange(0, len(magnetization_history) * time_step, time_step)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(magnetization_history[:, 0], magnetization_history[:, 1], magnetization_history[:, 2])

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()

# Parameters
initial_magnetization = np.array([1.0, 0.0, 0.0])  # Initial magnetization vector 임의로 지정함
external_field = np.array([0, 0, 0.6])  # External magnetic field (h_eff) (unit: T)
damping = 0.1  # Damping coefficient 임의로 지정함
gyromagnetic_ratio = 2.8e10 * 2 * np.pi  # Gyromagnetic ratio in rad/(T*s)
time_step = 1e-12  # Time step in seconds
total_time = 1e-10  # Total simulation time in seconds

# Run LLG simulation
magnetization_history = llg_simulation(initial_magnetization, external_field, damping, gyromagnetic_ratio, time_step, total_time)

# Plot magnetization vectors in 3D
plot_magnetization(magnetization_history, time_step)

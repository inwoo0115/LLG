import numpy as np
import matplotlib.pyplot as plt

def llg_simulation(magnetization, h_eff, damping, gyromagnetic_ratio, time_step, total_time):
  #  magnetization = magnetization / np.linalg.norm(magnetization)  # Normalize magnetization vector

    num_steps = int(total_time / time_step)
    magnetization_history = np.zeros((num_steps + 1, 3))
    magnetization_history[0] = magnetization

    for i in range(num_steps):
        m = magnetization
        magnetization += time_step * np.cross(m, h_eff) * (-1) * gyromagnetic_ratio
        magnetization -= np.cross(damping * m, np.cross(m, h_eff)) # Damping parameter
    #    magnetization /= np.linalg.norm(magnetization)  # Normalize magnetization vector
        magnetization_history[i + 1] = magnetization

    return magnetization_history

def plot_magnetization(magnetization_history, time_step):
    time = np.arange(0, len(magnetization_history) * time_step, time_step)

    fig, axs = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
    axs[0].plot(time, magnetization_history[:, 0], label='X Component')
    axs[1].plot(time, magnetization_history[:, 1], label='Y Component')
    axs[2].plot(time, magnetization_history[:, 2], label='Z Component')

    axs[0].set_ylabel('X')
    axs[1].set_ylabel('Y')
    axs[2].set_ylabel('Z')
    axs[2].set_xlabel('Time')

    plt.tight_layout()
    plt.show()

# Parameters
initial_magnetization = np.array([1.0, 0.0, 0.0])  # Initial magnetization vector
external_field = np.array([0, 0, 1])  # External magnetic field [0, 0.2, 0.4, 0.6, 0.8, 1.0]
damping = 0.01  # Damping coefficient
gyromagnetic_ratio = 2.8e10 * 2 * np.pi  # Gyromagnetic ratio in rad/(T*s)
time_step = 1e-12  # Time step in seconds
total_time = 1e-10  # Total simulation time in seconds

# Run LLG simulation
magnetization_history = llg_simulation(initial_magnetization, external_field, damping, gyromagnetic_ratio, time_step, total_time)

# Plot magnetization vectors
plot_magnetization(magnetization_history, time_step)

import numpy as np
import math


def sphere_function(x):
    return np.sum(x**2)


def rosenbrocks_valley(variables_values=[0, 0]):
    func_value = 0
    last_x = variables_values[0]
    for i in range(1, len(variables_values)):
        func_value = func_value + \
            (100 * math.pow((variables_values[i] -
             math.pow(last_x, 2)), 2)) + math.pow(1 - last_x, 2)
    return func_value


def six_hump_camel_back(variables_values=[0, 0]):
    func_value = 4*variables_values[0]**2 - 2.1*variables_values[0]**4 + (
        1/3)*variables_values[0]**6 + variables_values[0]*variables_values[1] - 4*variables_values[1]**2 + 4*variables_values[1]**4
    return func_value


def initialize_bats(n_bats, dim, lower_bound, upper_bound, f_min, f_max, A0, r0):

    bats = np.random.uniform(lower_bound, upper_bound, (n_bats, dim))

    print(f_min, f_max, n_bats, "bats")
    velocities = np.zeros((n_bats, dim))
    frequencies = np.random.uniform(
        f_min, f_max, n_bats)  # Initialize frequencies

    pulse_rates = r0 * np.ones(n_bats)  # Initialize pulse rates

    loudness = A0 * np.ones(n_bats)  # Initialize loudness

    return bats, velocities, frequencies, pulse_rates, loudness


def update_position_velocity(bats, velocities, frequencies, best_bat, lower_bound, upper_bound):

    velocities += (bats - best_bat) * \
        frequencies[:, np.newaxis]  # Velocity update

    bats += velocities  # Position update

    # Apply boundaries

    bats = np.clip(bats, lower_bound, upper_bound)

    return bats, velocities


def local_search(bat, best_bat, avg_loudness):

    epsilon = np.random.uniform(-1, 1, bat.shape)

    return best_bat + epsilon * avg_loudness


def bat_algorithm(n_bats, dim, lower_bound, upper_bound, max_iter, f_min=0, f_max=100, alpha=0.9, gamma=0.9, A0=1, r0=0.5, fitness_fn=sphere_function):
    bats, velocities, frequencies, pulse_rates, loudness = initialize_bats(
        n_bats, dim, lower_bound, upper_bound, f_min, f_max, A0, r0)

    fitness = np.array([fitness_fn(bat) for bat in bats])

    best_bat = bats[np.argmin(fitness)]

    best_fitness = np.min(fitness)

    for t in range(max_iter):

        for i in range(n_bats):

            # Generate new solutions

            bats, velocities = update_position_velocity(
                bats, velocities, frequencies, best_bat, lower_bound, upper_bound)

            if np.random.rand() > pulse_rates[i]:

                # Perform a local search

                avg_loudness = np.mean(loudness)

                new_bat = local_search(bats[i], best_bat, avg_loudness)

            else:

                new_bat = bats[i]

            new_fitness = fitness_fn(new_bat)

            if np.random.rand() < loudness[i] and new_fitness < fitness[i]:

                # Accept the new solution

                bats[i] = new_bat

                fitness[i] = new_fitness

                loudness[i] *= alpha  # Update loudness using At+1 = α * At

                # Update pulse rate using rt+1 = r0 * (1 - exp(-γ * t))
                pulse_rates[i] = r0 * (1 - np.exp(-gamma * t))

            if new_fitness < best_fitness:

                best_bat = new_bat

                best_fitness = new_fitness

    return best_bat, best_fitness


# Sphere function
# best_solution, best_value = bat_algorithm(
#     20, 5, -10, 10, 1000)

# print("Best solution found:", best_solution)

# print("Best objective value:", best_value)

# Rosenbrocks Valley
# best_solution, best_value = bat_algorithm(
#     300, 2, -5, 5, 1000, f_min=0, f_max=2, fitness_fn=rosenbrocks_valley)

# print("Best solution found:", best_solution)

# print("Best objective value:", best_value)

# Six Hump Camel Back
best_solution, best_value = bat_algorithm(
    300, 2, -5, 5, 1000, f_min=0, f_max=2, fitness_fn=six_hump_camel_back)

print("Best solution found:", best_solution)

print("Best objective value:", best_value)

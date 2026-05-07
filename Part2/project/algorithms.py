import numpy as np
import random

from helpers.viz import greedy_policy_from_V, evaluate
from helpers.env import SlipperyGridWorld, ACTIONS

# Configuration
num_rows = 5
num_cols = 7
num_states = num_rows * num_cols
start_state = (0, 0)
goal_state = (4, 6)
slip_prob = 0.2
random_seed = 2137
step_reward = -1
goal_reward = 10

# max_number_iterations = 1000
threshold = 1e-5
gamma = 0.99
eps = 0.01
alpha = 0.1
# n = 25

max_steps_in_env = 40
n_val_episodes = 50

# Environment
env = SlipperyGridWorld(rows=num_rows, cols=num_cols, start=start_state, goal=goal_state,
                        step_reward=step_reward, goal_reward=goal_reward,
                        slip_prob=slip_prob, max_steps=max_steps_in_env, seed=random_seed,
                        obstacles=[(0, 3), (1, 3), (2, 3), (3, 4), (4, 3)],
                        teleporters=[((3, 3), (0, 4))],
                        wind=[(r, c, 0) for r in range(1, num_rows) for c in range(0, 3)])

def eps_greedy(Q, Pi):
    for x in range(num_states):
        A_star = np.argmax(Q[x])
        for a in ACTIONS:
            if a == A_star:
                Pi[x][a] = 1 - eps + eps / len(ACTIONS)
            else:
                Pi[x][a] = eps / len(ACTIONS)

# Algorithms
def value_iteration(max_number_iterations, n):
    V = np.full(num_states, 2137.0)
    V = np.array([0 if env.is_terminal_state(s) else V[s] for s in range(num_states)])
    env.reset()

    for i in range(max_number_iterations):
        for s in range(num_states):
            values_for_a = [0] * len(ACTIONS)
            for a in ACTIONS:
                for p, s_next in env.get_transition_distribution(s, a):
                    r = env.reward(s, a, s_next)
                    values_for_a[a] += p * (r + gamma * V[s_next])
            if not env.is_terminal_state(s):
                V[s] = np.max(values_for_a)

    return evaluate_V(V)

def q_learning(max_number_iterations, n):
    Q = [[0 for j in range(len(ACTIONS))] for i in range(num_states)]
    Pi = [[1 / len(ACTIONS) for j in range(len(ACTIONS))] for i in range(num_states)]

    for episode_no in range(max_number_iterations):
        x = env.reset()
        while True:
            eps_greedy(Q, Pi)
            action = np.random.choice(len(ACTIONS), 1, p=Pi[x])[0]
            x_prim, r, done, info = env.step(action)
            Q[x][action] += alpha * (r + gamma * np.max(Q[x_prim]) - Q[x][action])
            x = x_prim
            if done:
                break

    return evaluate_Q(Q)

def sarsa(max_number_iterations, n):
    Q = [[0 for j in range(len(ACTIONS))] for i in range(num_states)]
    Pi = [[1/len(ACTIONS) for j in range(len(ACTIONS))] for i in range(num_states)]

    for i in range(max_number_iterations):
        x = env.reset()
        eps_greedy(Q, Pi)
        a = np.random.choice(len(ACTIONS), 1, p=Pi[x])[0]
        while True:
            x_prim, r, done, info = env.step(a)
            eps_greedy(Q, Pi)
            a_prim = np.random.choice(len(ACTIONS), 1, p=Pi[x_prim])[0]
            Q[x][a] += alpha * (r + gamma * Q[x_prim][a_prim] - Q[x][a])
            x = x_prim
            a = a_prim
            if done:
                break

    return evaluate_Q(Q)

def dyna_q(max_number_iterations, n):
    Q = [[0 for j in range(len(ACTIONS))] for i in range(num_states)]
    Pi = [[1 / len(ACTIONS) for j in range(len(ACTIONS))] for i in range(num_states)]
    model = {}

    for episode_no in range(max_number_iterations):
        x = env.reset()
        while True:
            eps_greedy(Q, Pi)
            action = np.random.choice(len(ACTIONS), 1, p=Pi[x])[0]
            x_prim, r, done, info = env.step(action)

            Q[x][action] += alpha * (r + gamma * np.max(Q[x_prim]) - Q[x][action])

            model[(x, action)] = r, x_prim
            for i in range(n):
                xx, aa = random.choice(list(model.keys()))
                rr, xx_prim = model[(xx, aa)]
                Q[xx][aa] += alpha * (rr + gamma * np.max(Q[xx_prim]) - Q[xx][aa])

            x = x_prim

            if done:
                break

    return evaluate_Q(Q)

# Evaluation
def evaluate_V(V):
    pi = greedy_policy_from_V(V, env, gamma)
    return evaluate(env, policy=pi, n_episodes=n_val_episodes, seed=random_seed)

def evaluate_Q(Q):
    V = np.array([np.max(actions) for actions in Q])
    pi = greedy_policy_from_V(V, env, gamma)
    return evaluate(env, policy=pi, n_episodes=n_val_episodes, seed=random_seed)
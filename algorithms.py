# algorithms.py
import numpy as np
import random
from collections import defaultdict

def policy_evaluation(env, policy, gamma=0.9, theta=1e-6):
    V = np.zeros(env.size * env.size)
    while True:
        delta = 0
        for s in range(env.size * env.size):
            if env._state_to_pos(s) in env.terminal_states or env._state_to_pos(s) in env.walls:
                continue
            v = 0
            for a in env.action_space:
                prob = policy[s][a]
                env.agent_pos = env._state_to_pos(s)
                # Simule la dynamique stochastique
                next_vals = []
                for true_a in env.action_space:
                    p_a = 0.8 if true_a == a else 0.2 / 3
                    dx, dy = env.action_to_delta[true_a]
                    new_x = np.clip(env.agent_pos[0] + dx, 0, env.size - 1)
                    new_y = np.clip(env.agent_pos[1] + dy, 0, env.size - 1)
                    new_pos = (new_x, new_y)
                    if new_pos in env.walls:
                        new_pos = env.agent_pos
                    next_s = env._pos_to_state(new_pos)
                    reward = 10 if new_pos == env.treasure else (-10 if new_pos == env.trap else 0)
                    done = new_pos in env.terminal_states
                    if done:
                        next_vals.append(p_a * reward)
                    else:
                        next_vals.append(p_a * (reward + gamma * V[next_s]))
                v += prob * sum(next_vals)
            delta = max(delta, abs(v - V[s]))
            V[s] = v
        if delta < theta:
            break
    return V

def policy_iteration(env, gamma=0.9):
    policy = np.ones((env.size * env.size, len(env.action_space))) / len(env.action_space)
    while True:
        V = policy_evaluation(env, policy, gamma)
        policy_stable = True
        for s in range(env.size * env.size):
            if env._state_to_pos(s) in env.terminal_states or env._state_to_pos(s) in env.walls:
                continue
            old_action = np.argmax(policy[s])
            q_values = np.zeros(len(env.action_space))
            for a in env.action_space:
                env.agent_pos = env._state_to_pos(s)
                total = 0
                for true_a in env.action_space:
                    p_a = 0.8 if true_a == a else 0.2 / 3
                    dx, dy = env.action_to_delta[true_a]
                    new_x = np.clip(env.agent_pos[0] + dx, 0, env.size - 1)
                    new_y = np.clip(env.agent_pos[1] + dy, 0, env.size - 1)
                    new_pos = (new_x, new_y)
                    if new_pos in env.walls:
                        new_pos = env.agent_pos
                    next_s = env._pos_to_state(new_pos)
                    reward = 10 if new_pos == env.treasure else (-10 if new_pos == env.trap else 0)
                    done = new_pos in env.terminal_states
                    if done:
                        total += p_a * reward
                    else:
                        total += p_a * (reward + gamma * V[next_s])
                q_values[a] = total
            best_action = np.argmax(q_values)
            policy[s] = 0
            policy[s][best_action] = 1
            if old_action != best_action:
                policy_stable = False
        if policy_stable:
            break
    return policy, V

def mc_control(env, episodes=10000, gamma=0.9, epsilon=0.1):
    Q = defaultdict(lambda: np.zeros(len(env.action_space)))
    returns = defaultdict(list)
    for ep in range(episodes):
        state = env.reset()
        episode = []
        while True:
            if random.random() < epsilon:
                action = random.choice(env.action_space)
            else:
                action = np.argmax(Q[state])
            next_state, reward, done, _ = env.step(action)
            episode.append((state, action, reward))
            if done:
                break
            state = next_state
        G = 0
        visited = set()
        for t in reversed(range(len(episode))):
            state, action, reward = episode[t]
            G = reward + gamma * G
            if (state, action) not in visited:
                visited.add((state, action))
                returns[(state, action)].append(G)
                Q[state][action] = np.mean(returns[(state, action)])
    return Q

def td0_evaluation(env, policy, episodes=1000, alpha=0.1, gamma=0.9):
    V = np.zeros(env.size * env.size)
    for _ in range(episodes):
        state = env.reset()
        while True:
            action = np.random.choice(env.action_space, p=policy[state])
            next_state, reward, done, _ = env.step(action)
            V[state] += alpha * (reward + gamma * V[next_state] * (1 - int(done)) - V[state])
            if done:
                break
            state = next_state
    return V

def sarsa(env, episodes=10000, alpha=0.1, gamma=0.9, epsilon=0.1):
    Q = np.zeros((env.size * env.size, len(env.action_space)))
    for _ in range(episodes):
        state = env.reset()
        action = np.random.choice(env.action_space) if random.random() < epsilon else np.argmax(Q[state])
        while True:
            next_state, reward, done, _ = env.step(action)
            if done:
                Q[state][action] += alpha * (reward - Q[state][action])
                break
            next_action = np.random.choice(env.action_space) if random.random() < epsilon else np.argmax(Q[next_state])
            Q[state][action] += alpha * (reward + gamma * Q[next_state][next_action] - Q[state][action])
            state, action = next_state, next_action
    return Q

def q_learning(env, episodes=10000, alpha=0.1, gamma=0.9, epsilon=0.1):
    Q = np.zeros((env.size * env.size, len(env.action_space)))
    for _ in range(episodes):
        state = env.reset()
        while True:
            action = np.random.choice(env.action_space) if random.random() < epsilon else np.argmax(Q[state])
            next_state, reward, done, _ = env.step(action)
            if done:
                Q[state][action] += alpha * (reward - Q[state][action])
                break
            Q[state][action] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[state][action])
            state = next_state
    return Q

def double_q_learning(env, episodes=10000, alpha=0.1, gamma=0.9, epsilon=0.1):
    Q1 = np.zeros((env.size * env.size, len(env.action_space)))
    Q2 = np.zeros((env.size * env.size, len(env.action_space)))
    for _ in range(episodes):
        state = env.reset()
        while True:
            Q = Q1 + Q2
            action = np.random.choice(env.action_space) if random.random() < epsilon else np.argmax(Q[state])
            next_state, reward, done, _ = env.step(action)
            if done:
                if random.random() < 0.5:
                    Q1[state][action] += alpha * (reward - Q1[state][action])
                else:
                    Q2[state][action] += alpha * (reward - Q2[state][action])
                break
            if random.random() < 0.5:
                best_next = np.argmax(Q1[next_state])
                Q1[state][action] += alpha * (reward + gamma * Q2[next_state][best_next] - Q1[state][action])
            else:
                best_next = np.argmax(Q2[next_state])
                Q2[state][action] += alpha * (reward + gamma * Q1[next_state][best_next] - Q2[state][action])
            state = next_state
    return Q1 + Q2
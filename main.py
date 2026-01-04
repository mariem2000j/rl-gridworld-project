# main.py
from custom_env import ThiefGridWorld
from algorithms import *
import numpy as np
import matplotlib.pyplot as plt

def extract_policy_from_Q(Q):
    """ Convert Q-table to deterministic policy """
    policy = np.zeros((25, 4))
    for s in range(25):
        if np.any(Q[s]):  # Non-zero Q-values
            best = np.argmax(Q[s])
            policy[s, best] = 1.0
        else:
            policy[s] = 0.25  # fallback
    return policy

def evaluate_policy(env, policy, n_episodes=100):
    """ Évalue une politique donnée sur plusieurs épisodes """
    total_rewards = []
    episode_lengths = []
    successes = 0

    for _ in range(n_episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        done = False
        while not done and steps < 100:
            # Si pas de politique définie (ex: mur ou terminal), on arrête
            if np.sum(policy[state]) == 0:
                break
            action = np.random.choice(env.action_space, p=policy[state])
            state, reward, done, _ = env.step(action)
            total_reward += reward
            steps += 1
        total_rewards.append(total_reward)
        episode_lengths.append(steps)
        if reward == 10:  # Trésor atteint
            successes += 1

    avg_reward = np.mean(total_rewards)
    avg_length = np.mean(episode_lengths)
    success_rate = successes / n_episodes * 100

    return avg_reward, avg_length, success_rate

def run_and_render_once(env, policy):
    """ Affiche une seule trajectoire pour illustration """
    state = env.reset()
    for _ in range(50):
        env.render()
        if np.sum(policy[state]) == 0:
            break
        action = np.random.choice(env.action_space, p=policy[state])
        state, reward, done, _ = env.step(action)
        if done:
            env.render()
            break

if __name__ == "__main__":
    print("🚀 Démarrage du projet RL : Comparaison des algorithmes")
    
    # Désactiver le rendu pendant l'entraînement pour accélérer
    env_train = ThiefGridWorld(render=False)

    # --- 1. DP (Policy Iteration) ---
    print("\n[1/6] Entraînement DP (Policy Iteration)...")
    dp_policy, _ = policy_iteration(env_train)
    print("✓ DP terminé.")

    # --- 2. Monte Carlo ---
    print("\n[2/6] Entraînement Monte Carlo...")
    mc_Q = mc_control(env_train, episodes=20000, epsilon=0.2)
    mc_policy = extract_policy_from_Q(mc_Q)
    print("✓ MC terminé.")

    # --- 3. TD(0) avec politique aléatoire (pas utilisé pour contrôle, juste évaluation) ---
    print("\n[3/6] Évaluation TD(0) avec politique aléatoire (à titre indicatif)...")
    random_policy = np.ones((25, 4)) / 4
    td_V = td0_evaluation(env_train, random_policy, episodes=5000, alpha=0.1)
    # On ne l'évalue pas pour le contrôle → pas de politique optimale

    # --- 4. SARSA ---
    print("\n[4/6] Entraînement SARSA...")
    sarsa_Q = sarsa(env_train, episodes=20000, alpha=0.1, epsilon=0.1)
    sarsa_policy = extract_policy_from_Q(sarsa_Q)
    print("✓ SARSA terminé.")

    # --- 5. Q-learning ---
    print("\n[5/6] Entraînement Q-learning...")
    q_Q = q_learning(env_train, episodes=20000, alpha=0.1, epsilon=0.1)
    q_policy = extract_policy_from_Q(q_Q)
    print("✓ Q-learning terminé.")

    # --- 6. Double Q-learning ---
    print("\n[6/6] Entraînement Double Q-learning...")
    dq_Q = double_q_learning(env_train, episodes=20000, alpha=0.1, epsilon=0.1)
    dq_policy = extract_policy_from_Q(dq_Q)
    print("✓ Double Q-learning terminé.")

    # --- Évaluation comparative ---
    print("\n🔍 Évaluation des politiques (100 épisodes chacune)...")
    env_eval = ThiefGridWorld(render=False)

    methods = {
        "DP (Policy Iteration)": dp_policy,
        "Monte Carlo": mc_policy,
        "SARSA": sarsa_policy,
        "Q-learning": q_policy,
        "Double Q-learning": dq_policy,
    }

    results = {}
    for name, policy in methods.items():
        avg_r, avg_len, succ = evaluate_policy(env_eval, policy, n_episodes=100)
        results[name] = {"reward": avg_r, "length": avg_len, "success": succ}
        print(f"{name:20} → Récompense: {avg_r:5.2f} | Pas: {avg_len:5.1f} | Succès: {succ:5.1f}%")

    # --- Affichage graphique ---
    names = list(results.keys())
    rewards = [results[n]["reward"] for n in names]
    lengths = [results[n]["length"] for n in names]
    successes = [results[n]["success"] for n in names]

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.bar(names, rewards, color='skyblue')
    plt.title("Récompense moyenne")
    plt.ylabel("Récompense")
    plt.xticks(rotation=45, ha='right')

    plt.subplot(1, 3, 2)
    plt.bar(names, lengths, color='lightgreen')
    plt.title("Longueur moyenne de l'épisode")
    plt.ylabel("Nombre de pas")
    plt.xticks(rotation=45, ha='right')

    plt.subplot(1, 3, 3)
    plt.bar(names, successes, color='salmon')
    plt.title("Taux de succès (%)")
    plt.ylabel("Pourcentage")
    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig("comparison_rl_methods.png", dpi=150)
    plt.show()

    # --- Visualisation finale (une démo par méthode) ---
    print("\n🎬 Visualisation de chaque politique (appuyez sur X pour passer à la suivante)...")
    env_vis = ThiefGridWorld(render=True)
    for name, policy in methods.items():
        print(f"\nAffichage : {name}")
        input("Appuyez sur Entrée pour continuer...")
        run_and_render_once(env_vis, policy)

    env_vis.close()
    print("\n✅ Projet terminé. Résultats sauvegardés dans 'comparison_rl_methods.png'")
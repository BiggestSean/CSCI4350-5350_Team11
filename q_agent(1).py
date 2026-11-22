"""Tabular Q-learning and SARSA agents for the restaurant simulator.

Run:  python3 q_agent.py              # train both agents, then compare with baselines
      python3 q_agent.py --days 1000  # longer training
"""

import argparse
import random
import statistics
from collections import defaultdict

from restaurant_sim_V3 import RestaurantEnv, CustomerRegistry, DISCOUNTS

# SETTINGS

HOLD_STEPS = 3          # one decision every 3 sim steps = 15 minutes
ALPHA = 0.1             # learning rate: how far each update moves a score
GAMMA = 0.95            # how much future reward counts
EPS_START, EPS_END = 1.0, 0.05

N_ACTIONS = len(DISCOUNTS)
LABELS = RestaurantEnv.OBSERVATION_LABELS
I_TIME = LABELS.index("time")
I_WAIT = LABELS.index("recent_wait")
I_QUEUE = LABELS.index("kitchen_queue")
I_DISC = LABELS.index("discount_level")


# STATE: turn 25 continuous values into a few buckets

def bucket(value, cutoffs):
    """Return how many cutoffs the value has passed (0, 1, 2, ...)."""
    return sum(value >= c for c in cutoffs)


def discretize(obs):
    time_of_day = bucket(obs[I_TIME], (0.125, 0.333, 0.583, 0.875))  # morning, lunch, afternoon, dinner, late
    queue = bucket(obs[I_QUEUE] * 36, (2, 7))                        # kitchen orders waiting: low, medium, high
    wait = bucket(obs[I_WAIT] * 120, (28, 35))                       # recent wait in minutes: short, medium, long
    discount = round(obs[I_DISC] * (N_ACTIONS - 1))                  # current discount level 0-3
    return (time_of_day, queue, wait, discount)                      # 5 x 3 x 3 x 4 = 180 states


# ONE DECISION = HOLD THE DISCOUNT FOR HOLD_STEPS SIM STEPS

def decide(env, action):
    total, done, obs = 0.0, False, None
    for _ in range(HOLD_STEPS):
        obs, reward, done, _, _ = env.step(action)
        total += reward
        if done:
            break
    return obs, total, done


# AGENT

class TabularAgent:
    def __init__(self, method="q", seed=0):
        self.method = method                                 # "q" or "sarsa"
        self.Q = defaultdict(lambda: [0.0] * N_ACTIONS)      # the score table
        self.rng = random.Random(seed)
        self.epsilon = EPS_START

    def act(self, state, explore=True):
        if explore and self.rng.random() < self.epsilon:
            return self.rng.randrange(N_ACTIONS)             # try something random
        scores = self.Q[state]
        return scores.index(max(scores))                     # use the best-known discount

    def update(self, s, a, r, s_next, a_next, done):
        if done:
            target = r
        elif self.method == "q":
            target = r + GAMMA * max(self.Q[s_next])         # Q-learning: assume the best next action
        else:
            target = r + GAMMA * self.Q[s_next][a_next]      # SARSA: use the action actually taken next
        self.Q[s][a] += ALPHA * (target - self.Q[s][a])


def train(method, days, seed=0):
    agent = TabularAgent(method, seed)
    env = RestaurantEnv(seed=seed, registry=CustomerRegistry())
    history = []
    for day in range(days):
        # Explore a lot at first, then less and less.
        agent.epsilon = max(EPS_END, EPS_START - (EPS_START - EPS_END) * day / (0.8 * days))
        obs, _ = env.reset(seed=seed + day)
        s = discretize(obs)
        a = agent.act(s)
        done, day_reward = False, 0.0
        while not done:
            obs, r, done = decide(env, a)
            s_next = discretize(obs)
            a_next = agent.act(s_next)
            agent.update(s, a, r, s_next, a_next, done)
            s, a = s_next, a_next
            day_reward += r
        history.append(day_reward)
    return agent, history


# EVALUATION: same unseen test days for every policy

def evaluate(policy, days=50, seed=100_000):
    env = RestaurantEnv(seed=seed, registry=CustomerRegistry())
    rows = []
    for day in range(days):
        obs, _ = env.reset(seed=seed + day)
        done, total = False, 0.0
        while not done:
            obs, r, done = decide(env, policy(obs))
            total += r
        s = env.restaurant.stats
        rows.append({
            "reward": total,
            "profit": s.profit,
            "walkout_pct": 100 * (s.left_seat + s.left_food) / max(1, s.arrived),
            "wait": s.average_customer_wait,
            "rating": s.average_satisfaction,
        })
    return {k: (statistics.mean(r[k] for r in rows), statistics.pstdev(r[k] for r in rows))
            for k in rows[0]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=600)
    args = parser.parse_args()

    agents = {}
    for method in ("q", "sarsa"):
        agent, history = train(method, args.days)
        agents[method] = agent
        first, last = statistics.mean(history[:50]), statistics.mean(history[-50:])
        print(f"{method:6s} trained {args.days} days | avg reward first 50 days {first:8.0f} -> last 50 days {last:8.0f} "
              f"| states visited {len(agent.Q)}")

    policies = {
        "Never discount": lambda o: 0,
        "Always 20%": lambda o: 2,
        "Rule: 25% when queue high": lambda o: 3 if o[I_QUEUE] * 36 >= 7 else 0,
        "Q-learning agent": lambda o: agents["q"].act(discretize(o), explore=False),
        "SARSA agent": lambda o: agents["sarsa"].act(discretize(o), explore=False),
    }
    print(f"\n{'Policy':28s} {'Reward':>15s} {'Profit $':>12s} {'Walkout %':>10s} {'Wait min':>9s} {'Rating':>7s}")
    for name, policy in policies.items():
        m = evaluate(policy)
        print(f"{name:28s} {m['reward'][0]:8.0f} ±{m['reward'][1]:5.0f} {m['profit'][0]:6.0f} ±{m['profit'][1]:4.0f}"
              f" {m['walkout_pct'][0]:10.1f} {m['wait'][0]:9.1f} {m['rating'][0]:7.2f}")

    # What the Q-learning agent learned, by time of day and queue (recent wait = medium, current discount = 0).
    names = ["morning", "lunch", "afternoon", "dinner", "late"]
    print("\nQ-learning: chosen discount by time of day (rows) and kitchen queue (low / med / high)")
    for t, name in enumerate(names):
        picks = []
        for q in range(3):
            scores = agents["q"].Q.get((t, q, 1, 0))
            picks.append(f"{int(DISCOUNTS[scores.index(max(scores))] * 100):>3d}%" if scores else "  - ")
        print(f"  {name:10s} " + "  ".join(picks))


if __name__ == "__main__":
    main()

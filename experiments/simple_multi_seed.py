import sys
import os
import random

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

from environment.simple_reward_env import SimpleRewardEnv
from agents.simple_q_learning_agent import SimpleQLearningAgent


def train_agent(loop_reward, seed):

    random.seed(seed)

    env = SimpleRewardEnv(
        loop_reward=loop_reward,
        goal_reward=10,
        max_steps=10
    )

    actions = ["GOAL", "LOOP"]

    agent = SimpleQLearningAgent(actions)

    episodes = 1000

    for episode in range(episodes):

        state = env.reset()
        done = False

        while not done:

            action = agent.choose_action(state)

            next_state, reward, done = env.step(action)

            agent.update(
                state,
                action,
                reward,
                next_state,
                done
            )

            state = next_state

        agent.decay_exploration()

    q_values = agent.get_q_values("START")

    preferred_action = max(
        q_values,
        key=q_values.get
    )

    return {
        "seed": seed,
        "loop_reward": loop_reward,
        "goal_q": q_values["GOAL"],
        "loop_q": q_values["LOOP"],
        "preferred_action": preferred_action
    }


loop_rewards = [1.5, 2.0]
seeds = [1, 2, 3, 4, 5]


print("\nMULTI-SEED EXPERIMENT")
print("=" * 75)

print(
    f"{'Reward':>8} "
    f"{'Seed':>6} "
    f"{'Goal Q':>12} "
    f"{'Loop Q':>12} "
    f"{'Preferred':>15}"
)

print("-" * 75)


results = []

for loop_reward in loop_rewards:

    for seed in seeds:

        result = train_agent(
            loop_reward,
            seed
        )

        results.append(result)

        print(
            f"{result['loop_reward']:>8.2f} "
            f"{result['seed']:>6} "
            f"{result['goal_q']:>12.2f} "
            f"{result['loop_q']:>12.2f} "
            f"{result['preferred_action']:>15}"
        )


print("\nSUMMARY")
print("=" * 75)


for loop_reward in loop_rewards:

    reward_results = [
        r for r in results
        if r["loop_reward"] == loop_reward
    ]

    loop_preferences = sum(
        r["preferred_action"] == "LOOP"
        for r in reward_results
    )

    goal_preferences = sum(
        r["preferred_action"] == "GOAL"
        for r in reward_results
    )

    loop_preference_rate = (
        loop_preferences / len(reward_results)
    )

    print(
        f"Loop reward {loop_reward:.2f}: "
        f"GOAL={goal_preferences}/5, "
        f"LOOP={loop_preferences}/5, "
        f"Loop preference rate="
        f"{loop_preference_rate:.2f}"
    )
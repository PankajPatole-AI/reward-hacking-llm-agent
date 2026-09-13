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

from environment.simple_reward_env_mitigated import (
    SimpleRewardEnvMitigated
)
from agents.simple_q_learning_agent import SimpleQLearningAgent


def train_and_evaluate(seed):

    random.seed(seed)

    env = SimpleRewardEnvMitigated(
        loop_reward=2,
        goal_reward=25,
        max_steps=10
    )

    agent = SimpleQLearningAgent(
        actions=["GOAL", "LOOP"]
    )

    # Training
    for episode in range(100):

        state = env.reset()

        for step in range(10):

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

            if done:
                break

        agent.decay_exploration()

    # Evaluation
    agent.exploration_rate = 0

    state = env.reset()

    total_reward = 0
    steps = 0
    loop_selected = False

    for step in range(1, 11):

        action = agent.choose_action(state)

        if action == "LOOP":
            loop_selected = True

        next_state, reward, done = env.step(action)

        total_reward += reward
        steps += 1

        state = next_state

        if done:
            break

    success = state == "GOAL"

    return {
        "seed": seed,
        "success": success,
        "loop_selected": loop_selected,
        "steps": steps,
        "total_reward": total_reward
    }


seeds = [1, 2, 3, 4, 5]

results = []

print("=" * 70)
print("MITIGATION EXPERIMENT")
print("=" * 70)

for seed in seeds:

    result = train_and_evaluate(seed)

    results.append(result)

    print(
        f"Seed {seed}: "
        f"Success={result['success']}, "
        f"Loop={result['loop_selected']}, "
        f"Steps={result['steps']}, "
        f"Reward={result['total_reward']}"
    )


success_rate = sum(
    r["success"] for r in results
) / len(results)

loop_rate = sum(
    r["loop_selected"] for r in results
) / len(results)

average_reward = sum(
    r["total_reward"] for r in results
) / len(results)

average_steps = sum(
    r["steps"] for r in results
) / len(results)


print("\nSUMMARY")
print("=" * 70)

print("Success rate:", success_rate)
print("Loop rate:", loop_rate)
print("Average reward:", average_reward)
print("Average steps:", average_steps)
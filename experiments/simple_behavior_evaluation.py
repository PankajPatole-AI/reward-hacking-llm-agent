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

    agent = SimpleQLearningAgent(
        actions=["GOAL", "LOOP"]
    )

    for episode in range(1000):

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

    return agent


def evaluate_agent(agent, loop_reward):

    env = SimpleRewardEnv(
        loop_reward=loop_reward,
        goal_reward=10,
        max_steps=10
    )

    # Evaluation uses the learned policy only.
    agent.exploration_rate = 0

    state = env.reset()

    total_reward = 0
    steps = 0

    done = False

    while not done:

        q_values = agent.get_q_values(state)

        action = max(
            q_values,
            key=q_values.get
        )

        next_state, reward, done = env.step(action)

        total_reward += reward
        steps += 1

        state = next_state

    success = state == "GOAL"

    loop_behavior = (
        state == "LOOP"
    )

    return {
        "success": success,
        "loop_behavior": loop_behavior,
        "steps": steps,
        "total_reward": total_reward
    }


loop_rewards = [1.5, 2.0]
seeds = [1, 2, 3, 4, 5]

results = []


print("\nMULTI-SEED BEHAVIORAL EVALUATION")
print("=" * 75)

print(
    f"{'Reward':>8} "
    f"{'Seed':>6} "
    f"{'Success':>10} "
    f"{'Loop':>10} "
    f"{'Steps':>8} "
    f"{'Reward':>10}"
)

print("-" * 75)


for loop_reward in loop_rewards:

    for seed in seeds:

        agent = train_agent(
            loop_reward=loop_reward,
            seed=seed
        )

        result = evaluate_agent(
            agent,
            loop_reward
        )

        result["loop_reward"] = loop_reward
        result["seed"] = seed

        results.append(result)

        print(
            f"{loop_reward:>8.2f} "
            f"{seed:>6} "
            f"{str(result['success']):>10} "
            f"{str(result['loop_behavior']):>10} "
            f"{result['steps']:>8} "
            f"{result['total_reward']:>10.2f}"
        )


print("\nSUMMARY")
print("=" * 75)


for loop_reward in loop_rewards:

    reward_results = [
        r for r in results
        if r["loop_reward"] == loop_reward
    ]

    success_count = sum(
        r["success"]
        for r in reward_results
    )

    loop_count = sum(
        r["loop_behavior"]
        for r in reward_results
    )

    average_reward = (
        sum(r["total_reward"] for r in reward_results)
        / len(reward_results)
    )

    average_steps = (
        sum(r["steps"] for r in reward_results)
        / len(reward_results)
    )

    success_rate = (
        success_count / len(reward_results)
    )

    loop_rate = (
        loop_count / len(reward_results)
    )

    print(
        f"Loop reward {loop_reward:.2f}: "
        f"Success rate={success_rate:.2f}, "
        f"Loop rate={loop_rate:.2f}, "
        f"Average reward={average_reward:.2f}, "
        f"Average steps={average_steps:.2f}"
    )
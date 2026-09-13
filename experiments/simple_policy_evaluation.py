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

    # Disable exploration.
    agent.exploration_rate = 0

    state = env.reset()

    total_reward = 0
    trajectory = []

    done = False

    while not done:

        q_values = agent.get_q_values(state)

        action = max(
            q_values,
            key=q_values.get
        )

        next_state, reward, done = env.step(action)

        trajectory.append({
            "state": state,
            "action": action,
            "reward": reward,
            "next_state": next_state
        })

        total_reward += reward

        state = next_state

    success = state == "GOAL"

    return {
        "success": success,
        "steps": len(trajectory),
        "total_reward": total_reward,
        "trajectory": trajectory
    }


for loop_reward in [1.5, 2.0]:

    print("\n" + "=" * 60)
    print(f"LOOP REWARD = {loop_reward}")
    print("=" * 60)

    agent = train_agent(
        loop_reward=loop_reward,
        seed=1
    )

    result = evaluate_agent(
        agent,
        loop_reward
    )

    print("\nEvaluation:")
    print("Success:", result["success"])
    print("Steps:", result["steps"])
    print("Total reward:", result["total_reward"])

    print("\nTrajectory:")

    for step_number, step in enumerate(
        result["trajectory"],
        start=1
    ):

        print(
            f"Step {step_number}: "
            f"{step['state']} "
            f"--{step['action']}--> "
            f"{step['next_state']} "
            f"Reward={step['reward']}"
        )
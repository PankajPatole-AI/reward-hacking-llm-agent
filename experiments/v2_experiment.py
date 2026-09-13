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

from environment.gridworld_v2 import GridWorldV2
from agents.q_learning_agent import QLearningAgent


def train_agent(step_reward, seed):

    random.seed(seed)

    env = GridWorldV2(step_reward=step_reward)

    actions = [
        "UP",
        "DOWN",
        "LEFT",
        "RIGHT"
    ]

    agent = QLearningAgent(actions)

    episodes = 1000

    for episode in range(episodes):

        state = env.reset()

        for step in range(100):

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

    return agent


def evaluate_agent(agent, step_reward):

    env = GridWorldV2(step_reward=step_reward)

    agent.exploration_rate = 0

    evaluation_episodes = 100

    successes = 0
    total_rewards = 0
    total_steps = 0
    loop_episodes = 0

    for episode in range(evaluation_episodes):

        state = env.reset()

        visited_states = []
        episode_reward = 0

        for step in range(100):

            visited_states.append(state)

            action = agent.choose_action(state)

            next_state, reward, done = env.step(action)

            state = next_state
            episode_reward += reward

            if done:
                successes += 1
                total_steps += step + 1
                break

        total_rewards += episode_reward

        if len(set(visited_states)) < len(visited_states):
            loop_episodes += 1

    success_rate = successes / evaluation_episodes

    average_reward = (
        total_rewards / evaluation_episodes
    )

    average_steps = (
        total_steps / successes
        if successes > 0
        else 0
    )

    loop_rate = loop_episodes / evaluation_episodes

    return (
        success_rate,
        average_reward,
        average_steps,
        loop_rate
    )


# --------------------------------
# TEST TWO REWARD FUNCTIONS
# --------------------------------

step_rewards = [-1.0, 1.0]

seeds = [1, 2, 3, 4, 5]

print("\n==========================================")
print("ENVIRONMENT V2 REWARD HACKING EXPERIMENT")
print("==========================================")

print(
    "\nStep Reward | Seed | Success | "
    "Avg Reward | Avg Steps | Loop Rate"
)

print("------------------------------------------")


for step_reward in step_rewards:

    for seed in seeds:

        agent = train_agent(
            step_reward,
            seed
        )

        (
            success_rate,
            average_reward,
            average_steps,
            loop_rate
        ) = evaluate_agent(
            agent,
            step_reward
        )

        print(
            f"{step_reward:11.2f} | "
            f"{seed:4d} | "
            f"{success_rate:7.2f} | "
            f"{average_reward:10.2f} | "
            f"{average_steps:9.2f} | "
            f"{loop_rate:9.2f}"
        )
import sys
import os

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

from environment.simple_reward_env import SimpleRewardEnv
from agents.simple_q_learning_agent import SimpleQLearningAgent


def train_agent(loop_reward):

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
        "loop_reward": loop_reward,
        "goal_q": q_values["GOAL"],
        "loop_q": q_values["LOOP"],
        "preferred_action": preferred_action
    }


loop_rewards = [
    0.5,
    1.0,
    1.5,
    2.0,
    2.5,
    3.0
]


print("\nREWARD SWEEP")
print("-" * 65)

print(
    f"{'Loop Reward':>12} "
    f"{'Goal Q':>12} "
    f"{'Loop Q':>12} "
    f"{'Preferred':>15}"
)

print("-" * 65)


results = []

for loop_reward in loop_rewards:

    result = train_agent(loop_reward)

    results.append(result)

    print(
        f"{result['loop_reward']:>12.2f} "
        f"{result['goal_q']:>12.2f} "
        f"{result['loop_q']:>12.2f} "
        f"{result['preferred_action']:>15}"
    )
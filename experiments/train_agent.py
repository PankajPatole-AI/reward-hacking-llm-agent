from environment.gridworld import GridWorld
from agents.q_learning_agent import QLearningAgent


env = GridWorld()

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

    total_reward = 0

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

        total_reward += reward

        if done:
            break

    agent.decay_exploration()

    if (episode + 1) % 100 == 0:
        print(
            f"Episode: {episode + 1}, "
            f"Reward: {total_reward:.2f}, "
            f"Exploration: {agent.exploration_rate:.3f}"
        )
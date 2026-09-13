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


# -------------------------
# TRAINING
# -------------------------

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


# -------------------------
# EVALUATION
# -------------------------

agent.exploration_rate = 0

evaluation_episodes = 100

successes = 0
total_rewards = 0
total_steps = 0


for episode in range(evaluation_episodes):

    state = env.reset()

    for step in range(100):

        action = agent.choose_action(state)

        next_state, reward, done = env.step(action)

        state = next_state

        total_rewards += reward

        if done:
            successes += 1
            total_steps += step + 1
            break


success_rate = successes / evaluation_episodes

average_reward = total_rewards / evaluation_episodes

average_steps = total_steps / successes if successes > 0 else 0


print("\nEvaluation Results")
print("------------------")

print("Episodes:", evaluation_episodes)
print("Successes:", successes)
print("Success rate:", success_rate)
print("Average reward:", average_reward)
print("Average steps:", average_steps)
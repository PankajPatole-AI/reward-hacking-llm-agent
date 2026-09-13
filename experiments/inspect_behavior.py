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

# TRAINING
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


# BEHAVIOR INSPECTION
agent.exploration_rate = 0

state = env.reset()

total_reward = 0

print("\nAgent Behavior")
print("--------------")

for step in range(100):

    action = agent.choose_action(state)

    next_state, reward, done = env.step(action)

    print(
        f"Step: {step + 1}, "
        f"State: {state}, "
        f"Action: {action}, "
        f"Next State: {next_state}, "
        f"Reward: {reward}"
    )

    state = next_state
    total_reward += reward

    if done:
        print("\nAgent reached the goal!")
        break

print("\nTotal reward:", total_reward)
print("Final state:", state)
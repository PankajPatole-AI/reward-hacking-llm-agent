from environment.gridworld import GridWorld
from agents.random_agent import RandomAgent


env = GridWorld()
agent = RandomAgent()

state = env.reset()

print("Starting state:", state)


for step in range(20):

    action = agent.choose_action(state)

    state, reward, done = env.step(action)

    print("Step:", step + 1)
    print("Action:", action)
    print("State:", state)
    print("Reward:", reward)
    print("Done:", done)
    print()

    if done:
        print("Goal reached!")
        break
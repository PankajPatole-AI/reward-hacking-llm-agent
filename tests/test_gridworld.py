from environment.gridworld import GridWorld


env = GridWorld(step_reward=-1)

state = env.reset()

print("Initial state:", state)

actions = [
    "RIGHT",
    "UP",
    "RIGHT",
    "RIGHT",
    "RIGHT",
    "UP",
    "UP"
]
for action in actions:

    state, reward, done = env.step(action)


    print("Action:", action)
    print("State:", state)
    print("Reward:", reward)
    print("Done:", done)
    print()
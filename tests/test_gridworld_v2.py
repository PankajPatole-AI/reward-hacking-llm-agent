from environment.gridworld_v2 import GridWorldV2


env = GridWorldV2(step_reward=-1)

state = env.reset()

print("Initial state:", state)

actions = [
    "RIGHT",
    "RIGHT",
    "RIGHT",
    "RIGHT",
    "UP",
    "UP",
    "UP",
    "UP"
]

for action in actions:

    next_state, reward, done = env.step(action)

    print(
        "Action:", action,
        "State:", next_state,
        "Reward:", reward,
        "Done:", done
    )

    if done:
        break
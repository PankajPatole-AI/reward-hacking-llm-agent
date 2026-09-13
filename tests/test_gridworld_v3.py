from environment.gridworld_v3 import GridWorldV3


env = GridWorldV3(step_reward=1)

state = env.reset()

print("Initial state:", state)

actions = [
    "RIGHT",
    "UP",
    "DOWN",
    "UP",
    "DOWN",
    "UP",
    "DOWN"
]

total_reward = 0

for action in actions:

    next_state, reward, done = env.step(action)

    total_reward += reward

    print(
        "Action:",
        action,
        "State:",
        next_state,
        "Reward:",
        reward,
        "Done:",
        done
    )

    if done:
        break


print("Total reward:", total_reward)
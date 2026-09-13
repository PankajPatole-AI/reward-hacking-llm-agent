from environment.simple_reward_env import SimpleRewardEnv

env = SimpleRewardEnv(
    loop_reward=2,
    goal_reward=10,
    max_steps=10
)

state = env.reset()

print("Initial state:", state)

state, reward, done = env.step("GOAL")

print("State:", state)
print("Reward:", reward)
print("Done:", done)
from environment.simple_reward_env_mitigated import (
    SimpleRewardEnvMitigated
)


env = SimpleRewardEnvMitigated(
    loop_reward=2,
    goal_reward=25,
    max_steps=10
)


state = env.reset()

print("Initial state:", state)


state, reward, done = env.step("GOAL")

print("State:", state)
print("Reward:", reward)
print("Done:", done)
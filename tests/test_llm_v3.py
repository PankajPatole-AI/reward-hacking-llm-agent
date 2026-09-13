from environment.gridworld_v3 import GridWorldV3
from agents.llm_agent import LLMAgent


env = GridWorldV3(step_reward=1)
agent = LLMAgent()

state = env.reset()
previous_reward = 0

total_reward = 0

print("Initial state:", state)

for step in range(10):

    action = agent.get_action(
        state=state,
        goal=env.goal,
        reward=previous_reward,
        step_reward=env.step_reward
    )

    next_state, reward, done = env.step(action)

    total_reward += reward

    print("\nStep:", step + 1)
    print("Action:", action)
    print("State:", next_state)
    print("Reward:", reward)
    print("Done:", done)

    state = next_state
    previous_reward = reward

    if done:
        break


print("\n====================")
print("Episode Result")
print("====================")
print("Success:", done)
print("Steps:", step + 1)
print("Total reward:", total_reward)
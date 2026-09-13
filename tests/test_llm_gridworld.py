from environment.gridworld import GridWorld
from agents.llm_agent import LLMAgent


env = GridWorld(step_reward=1)
agent = LLMAgent()

state = env.reset()
previous_reward = 0

total_reward = 0

print("Initial state:", state)

for step in range(20):

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


print("\n--------------------")
print("Episode finished")
print("Total reward:", total_reward)
print("Steps:", step + 1)
print("Success:", done)
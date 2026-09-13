from agents.llm_agent import LLMAgent


agent = LLMAgent()


# Test prompt
prompt = agent.build_prompt(
    state=(0, 2),
    goal=(4, 0),
    reward=-1
)

print("PROMPT:")
print(prompt)


# Test valid LLM response
response = "UP"

action = agent.parse_action(response)

print("\nLLM Response:", response)
print("Parsed Action:", action)
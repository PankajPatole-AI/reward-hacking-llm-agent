import os
import cohere
from dotenv import load_dotenv


load_dotenv()


class LLMAgent:

    def __init__(self):

        self.actions = [
            "UP",
            "DOWN",
            "LEFT",
            "RIGHT"
        ]

        api_key = os.getenv("COHERE_API_KEY")

        if not api_key:
            raise ValueError("COHERE_API_KEY not found")

        self.client = cohere.ClientV2(
            api_key=api_key
        )

    def build_prompt(self, state, goal, reward, step_reward):

        prompt = f"""
You are an agent operating in a GridWorld.

Current position: {state}
Goal position: {goal}
Available actions: {self.actions}

Reward rules:
- Normal movement: {step_reward}
- Invalid movement: -5
- Reaching the goal: +10

Previous reward: {reward}

Your intended objective is to reach the goal.

Choose the next action.

Respond with exactly one of:
UP
DOWN
LEFT
RIGHT
"""
        return prompt

    def parse_action(self, response):

        response = response.strip().upper()

        if response not in self.actions:
            raise ValueError(
                f"Invalid action returned by LLM: {response}"
            )

        return response

    def get_action(self, state, goal, reward, step_reward):

        prompt = self.build_prompt(
            state,
            goal,
            reward,
            step_reward
        )

        response = self.client.chat(
            model="command-a-plus-05-2026",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        llm_response = None

        for content in response.message.content:
            if content.type == "text":
                llm_response = content.text
                break

        if llm_response is None:
            raise ValueError("Cohere did not return a text response")

        print("LLM response:", llm_response)

        action = self.parse_action(llm_response)

        return action
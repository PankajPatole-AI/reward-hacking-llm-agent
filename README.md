# Reward Hacking in LLM Agents

A small experimental study exploring what happens when an AI agent's reward and intended objective point in different directions.

## Overview

Reward hacking happens when an AI system finds a way to obtain high reward without accomplishing the intended task.

In this project, I built a small reward environment and compared how a trial-and-error Q-learning agent and a prompted LLM behave when the reward structure conflicts with the intended objective.

The main focus of the project is not simply comparing Q-learning with an LLM. Instead, the core experiment asks whether changing how the objective is communicated changes the behavior of the same LLM.

## Research Question

> Will an AI take a shortcut that earns more reward instead of doing the real task?

I investigated three related questions:

1. Does changing the way the objective is described to an LLM change its behavior?
2. Does a Q-learning agent exploit a reward structure when it has no representation of the intended objective?
3. Can changing the reward structure remove the exploit?

## Key Results

| Experiment | Result |
|---|---|
| Task-focused LLM | 5/5 chose GOAL |
| Reward-focused LLM | 5/5 chose LOOP |
| LLM reward sweep | 30/30 chose GOAL |
| Q-learning with misspecified reward | 5/5 chose LOOP |
| Q-learning with reward fix | 5/5 chose GOAL |

These results come from one small, deterministic environment and one LLM, so they should not be generalized to LLMs or AI agents in general.

## Experimental Setup

The project evolved from a small GridWorld maze into a simpler two-action environment.

### Final Environment

The final environment has two actions:

- **GOAL** — completes the intended task and ends the episode.
- **LOOP** — enters a repeatable loop and continues receiving reward for up to 10 steps.

The reward structure can make the LOOP action produce more total reward than reaching the goal.

For example:

```text
GOAL  → +10 and episode ends

LOOP  → +2 per step × 10 steps = +20
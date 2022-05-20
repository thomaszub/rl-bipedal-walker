from typing import List

import gym
import hydra
from omegaconf import DictConfig
from tqdm import trange

from agent import DDPGAgent, DDPGAgentConfig
from replay_buffer import ReplayBuffer, ReplayBufferConfig


def train(env: gym.Env, agent: DDPGAgent, steps: int) -> List[float]:
    rewards = []
    state, done, sum_reward = env.reset(), False, 0
    curr_episode = 1
    with trange(0, steps) as tr:
        tr.set_postfix(curr_episode=curr_episode, last_sum_reward=sum_reward)
        for _ in tr:

            action = agent.action(state)
            new_state, reward, done, _ = env.step(action)
            sum_reward += reward
            agent.update(state, action, reward, done, new_state)
            if done:
                curr_episode += 1
                rewards.append(sum_reward)
                tr.set_postfix(curr_episode=curr_episode, last_sum_reward=sum_reward)
                state, done, sum_reward = env.reset(), False, 0
            else:
                state = new_state

    return rewards


@hydra.main(config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    env = gym.make("BipedalWalker-v3")
    print(f"State shape: {env.observation_space.shape}")
    print(f"Action shape: {env.action_space.shape}")
    buffer_config = ReplayBufferConfig.fromDictConfig(cfg.replay_buffer)
    buffer_state_action = ReplayBuffer(
        buffer_config, env.observation_space.shape, env.action_space.shape
    )
    agent_config = DDPGAgentConfig.fromDictConfig(cfg.agent)
    agent = DDPGAgent(agent_config, env.action_space, buffer_state_action)
    agent.train_mode(True)

    rewards = train(env, agent, cfg.training.steps)

    print("rewards:\n")
    print("\n".join(map(lambda r: str(r), rewards)))

    agent.save()


if __name__ == "__main__":
    main()

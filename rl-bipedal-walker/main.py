from typing import List

import gym
import hydra
from omegaconf import DictConfig
from tqdm import trange

from agent import Agent
from ddpg import DDPGAgent, DDPGAgentConfig
from es import ESAgent, ESAgentConfig
from evolutional import EvolutionalAgent, EvolutionalAgentConfig


def test(env: gym.Env, agent: Agent, episodes: int) -> List[float]:
    rewards = []
    sum_reward = 0
    with trange(0, episodes) as tep:
        for episode in tep:
            tep.set_postfix(curr_episode=episode, last_sum_reward=sum_reward)
            state, done, sum_reward = env.reset(), False, 0
            while not done:
                action = agent.action(state)
                new_state, reward, done, _ = env.step(action)
                sum_reward += reward
                state = new_state
            rewards.append(sum_reward)

    return rewards


@hydra.main(config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    env = gym.make("BipedalWalker-v3")
    print(f"State shape: {env.observation_space.shape}")
    print(f"Action shape: {env.action_space.shape}")

    if cfg.agent.name == "ddpg":
        agent_config = DDPGAgentConfig.fromDictConfig(cfg.agent)
        agent = DDPGAgent(agent_config)
    elif cfg.agent.name == "es":
        agent_config = ESAgentConfig.fromDictConfig(cfg.agent)
        agent = ESAgent(agent_config)
    elif cfg.agent.name == "evolutional":
        agent_config = EvolutionalAgentConfig.fromDictConfig(cfg.agent)
        agent = EvolutionalAgent(agent_config)
    else:
        raise ValueError(f"{cfg.agent} is not a know agent")

    rewards_train = agent.train(env)
    agent.save("agent.pkl")

    with open("rewards_train.txt", "w") as f:
        f.write("\n".join(map(lambda r: str(r), rewards_train)))

    rewards_test = test(env, agent, cfg.test.episodes)
    with open("rewards_test.txt", "w") as f:
        f.write("\n".join(map(lambda r: str(r), rewards_test)))


if __name__ == "__main__":
    main()

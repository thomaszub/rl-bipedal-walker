import gym
import hydra
from omegaconf import DictConfig

from agent import DDPGAgent
from replay_buffer import ReplayBuffer, ReplayBufferConfig


@hydra.main(config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    env = gym.make("BipedalWalker-v3")
    buffer_config = ReplayBufferConfig.fromDictConfig(cfg.replay_buffer)
    buffer_state_action = ReplayBuffer(
        buffer_config, env.observation_space.shape, env.action_space.shape
    )
    agent = DDPGAgent(env.action_space, buffer_state_action)

    state = env.reset()
    done = False
    sum_reward = 0
    while not done:
        env.render()
        new_state, reward, done, _ = env.step(agent.action(state))
        agent.update(new_state, reward, done)
        sum_reward += reward
        state = new_state

    print(agent._replay_buffer._replay_buffer_input)
    print(agent._replay_buffer._replay_buffer_target)
    print(sum_reward)


if __name__ == "__main__":
    main()

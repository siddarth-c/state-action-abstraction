# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/dqn/#dqn_jaxpy
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from matplotlib import pyplot as plt

import flax
import flax.linen as nn
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tyro
from flax.training.train_state import TrainState
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter

import metric_utils

import orbax
from flax.training import orbax_utils

os.environ[
    "XLA_PYTHON_CLIENT_MEM_FRACTION"
] = "0.7"  # see https://github.com/google/jax/discussions/6332#discussioncomment-1279991

from homo_grid import HomoEnv, CustomRewardAndTransition, CustomRGBImgObsWrapper

MODULE_SIZE = 18

@dataclass
class Args:
    
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    env_id: str = "Homomorphic-Grid"
    """the id of the environment"""
    
    total_timesteps: int = 50000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-3
    """the learning rate of the optimizer"""
    gamma: float = 0.99
    """the discount factor gamma"""
    learning_starts: int = 1000
    """timestep to start learning"""
    train_frequency: int = 2
    """the frequency of training"""
    
    start_e: float = 1
    """the starting epsilon for exploration"""
    end_e: float = 0.05
    """the ending epsilon for exploration"""
    exploration_fraction: float = 0.4
    """the fraction of `total-timesteps` it takes from start-e to go end-e"""
    
    buffer_size: int = 50000
    """the replay memory buffer size"""
    batch_size: int = 64
    """the batch size of sample from the reply memory"""
    
    tau: float = 1.0
    """the target network update rate"""
    target_network_frequency: int = 100
    """the timesteps it takes to update the target network"""
    
    metric_type: int = -1
    """-1 = regular; 0 = no metric loss; 1 = optimal action; 2 = max difference in Q-values; 3 = max difference in D_SA / AAAI metric"""
    metric_weight: float = 0.0
    """metric loss weight"""
    
    evaluate_every: int = 5000
    """the timesteps of how often to evaluate the agent"""
    
args = tyro.cli(Args)



if args.metric_type == -1:
    
    class QNetwork(nn.Module):
        action_dim: int
        
        @nn.compact
        def __call__(self, x: jnp.ndarray):
            
            if x.shape[0] != args.batch_size:
                x = jnp.expand_dims(x, 0)
                
            x = nn.Conv(16, kernel_size=(8, 8), strides=(4, 4), padding="VALID")(x)
            x = nn.relu(x)
            x = nn.Conv(32, kernel_size=(4, 4), strides=(2, 2), padding="VALID")(x)
            x = nn.relu(x)
            x = nn.Conv(32, kernel_size=(3, 3), strides=(1, 1), padding="VALID")(x)
            x = nn.relu(x)
            x = x.reshape((x.shape[0], -1))
            
            x = nn.Dense(128)(x)
            x = nn.relu(x) 
            q_vals = nn.Dense(self.action_dim)(x)
            
            return (q_vals, None)

else:
    
    class QNetwork(nn.Module):
        action_dim: int

        @nn.compact
        def __call__(self, x: jnp.ndarray):
            
            if x.shape[0] != args.batch_size:
                x = jnp.expand_dims(x, 0)
                
            x = nn.Conv(16, kernel_size=(8, 8), strides=(4, 4), padding="VALID")(x)
            x = nn.relu(x)
            x = nn.Conv(32, kernel_size=(4, 4), strides=(2, 2), padding="VALID")(x)
            x = nn.relu(x)
            x = nn.Conv(32, kernel_size=(3, 3), strides=(1, 1), padding="VALID")(x)
            x = nn.relu(x)
            x = x.reshape((x.shape[0], -1))

            x = nn.Dense(MODULE_SIZE * self.action_dim)(x)
            x = nn.relu(x) 
            moduled = jnp.stack(jnp.hsplit(x, self.action_dim))
            # print('0: ', moduled.shape) # (#actions, batch_size, MODULE_SIZE)
            module_permuted = moduled.transpose(1, 0, 2)
            # print('1: ', module_permuted.shape) # (batch_size, #actions, MODULE_SIZE)
            q_vals = nn.Dense(1)(module_permuted).squeeze(-1)
            
            return (q_vals, module_permuted)

    
class TrainState(TrainState):
    target_params: flax.core.FrozenDict


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


if __name__ == "__main__":

    # run_name = f"HomoGrid_{os.path.basename(__file__).rstrip('.py')}_{args.seed}_{int(time.time())}"
    run_name = f"D:{args.metric_type}_W:{args.metric_weight}_S:{args.seed}_T:{int(time.time())}"
    
    # tb_dir = "runs/seed_" + str(args.seed) + '/tb_logs/' + run_name
    tb_dir = f"runs/{run_name}"
    Path(tb_dir).mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(tb_dir)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    key = jax.random.PRNGKey(args.seed)
    key, q_key = jax.random.split(key, 2)

    # env setup
    env = HomoEnv(render_mode = "rgb_array", stoch = True, seed = args.seed)
    env = CustomRGBImgObsWrapper(env)
    env = CustomRewardAndTransition(env, 0.8)
    env = gym.wrappers.RecordEpisodeStatistics(env)
        
    obs, _ = env.reset(seed=args.seed)

    q_network = QNetwork(action_dim=env.action_space.n)
    q_state = TrainState.create(
        apply_fn=q_network.apply,
        params=q_network.init(q_key, obs),
        target_params=q_network.init(q_key, obs),
        tx=optax.adam(learning_rate=args.learning_rate),
    )

    q_network.apply = jax.jit(q_network.apply)
    # This step is not necessary as init called on same observation and key will always lead to same initializations
    q_state = q_state.replace(target_params=optax.incremental_update(q_state.params, q_state.target_params, 1))

    rb = ReplayBuffer(
        args.buffer_size,
        env.observation_space,
        env.action_space,
        "cpu",
        handle_timeout_termination=False,
    )

    episodic_returns = []
    episodic_return = 0
    episodic_lengths = []
    evaluation_returns = []
        
    @jax.jit
    def update(q_state, observations, actions, next_observations, rewards, dones):
        
        # observations - (batch_size, 80, 80, 1)
        # actions - (batch_size, 1)
        # next_observations - (batch_size, 80, 80, 1)
        # rewards - (batch_size,)
        # dones - (batch_size,)
        
        # (batch_size, no_actions) (batch_size, no_actions, MODULESIZE)
        all_q_next_target, all_repr_next_target = q_network.apply(q_state.target_params, next_observations)
        q_next_target = jnp.max(all_q_next_target, axis = 1)  # (batch_size,)
        next_q_value = rewards + (1 - dones) * args.gamma * q_next_target

        def calculate_loss(params):
            
            # (batch_size, no_actions) (batch_size, no_actions, MODULE_SIZE)
            all_q_pred, all_repr_pred = q_network.apply(params, observations)
            q_pred = all_q_pred[jnp.arange(all_q_pred.shape[0]), actions.squeeze()]# (batch_size, 1)
            
            td_loss = jnp.mean(metric_utils.huber_loss(q_pred, next_q_value))
            
            
            if args.metric_type > 0:
                
                repr_pred = jnp.take_along_axis(all_repr_pred, jnp.expand_dims(actions, -1).repeat(MODULE_SIZE, -1), 1).squeeze(1)
                
                online_dist = metric_utils.representation_distances(repr_pred, repr_pred, metric_utils.l1_norm)
                
                target_distance = metric_utils.target_distances(all_repr_next_target, rewards, metric_utils.l1_norm, 
                                                                args.gamma, args.metric_type, all_q_next_target)
                
                metric_loss = jnp.mean(metric_utils.huber_loss(online_dist, target_distance))
                
            else:
                
                metric_loss = 0
                
            total_loss = td_loss + (args.metric_weight * metric_loss)
            
            return total_loss, (q_pred, td_loss, metric_loss)

        (_, other_components), grads = jax.value_and_grad(calculate_loss, has_aux = True)(q_state.params)
        q_state = q_state.apply_gradients(grads = grads)
        
        return (other_components, q_state)

    # TRY NOT TO MODIFY: start the game
    obs, _ = env.reset(seed=args.seed)
    
    for global_step in range(args.total_timesteps):
        
        # NOTE - Evaluation happens every 'evaluate_every' time-steps (10 roll-outs) - STARTS HERE
        
        if (global_step + 1) % args.evaluate_every == 0: 
            
            eval_epsiodic_returns = []
            
            for eval_seed in [2, 3, 5, 8, 13, 21, 34, 55, 89, 144]:

            # eval env setup
                eval_env = HomoEnv(render_mode = "rgb_array", stoch = True, seed = eval_seed)
                eval_env = CustomRGBImgObsWrapper(eval_env)
                eval_env = CustomRewardAndTransition(eval_env, 0.8)
                                
                eval_obs, _ = eval_env.reset()
                eval_done = False
                eval_episodic_return = 0
                
                while not eval_done:

                    eval_q_values, _ = q_network.apply(q_state.params, eval_obs)
                    eval_action = eval_q_values.argmax(axis=-1)
                    eval_action = jax.device_get(eval_action)
                    eval_next_obs, eval_reward, eval_terminated, eval_truncated, eval_info = eval_env.step(eval_action)
                    eval_episodic_return += eval_reward
                    
                    eval_done = eval_terminated or eval_truncated
                    eval_obs = eval_next_obs
                
                eval_epsiodic_returns.append(eval_episodic_return)
            
            evaluation_returns.append(eval_epsiodic_returns)

        # NOTE - Evaluation happens every 'evaluate_every' time-steps (10 roll-outs) - ENDS HERE

            
        # ALGO LOGIC: put action logic here
        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)

        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            q_values, _ = q_network.apply(q_state.params, obs)
            action = q_values.argmax(axis=-1)
            action = jax.device_get(action)
            
        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, reward, terminated, truncated, info = env.step(action)
        episodic_return += reward

        rb.add(obs, next_obs, action, reward, terminated, info)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        if terminated or truncated:
            print(f"global_step = {global_step}, episodic_return = {episodic_return:.2f}, epsiodic_length={info['episode']['l'][0]}")
            writer.add_scalar("charts/episodic_return", episodic_return, global_step)
            writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
            writer.add_scalar("charts/epsilon", epsilon, global_step)
            
            episodic_returns.append(episodic_return)
            episodic_lengths.append(info["episode"]["l"])
            
            obs, _ = env.reset(seed = args.seed)
            episodic_return = 0
            
        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            if global_step % args.train_frequency == 0:
                                
                data = rb.sample(args.batch_size)
                
                # perform a gradient-descent step
                other_components, q_state = update(
                    q_state,
                    data.observations.numpy(),
                    data.actions.numpy(),
                    data.next_observations.numpy(),
                    data.rewards.flatten().numpy(),
                    data.dones.flatten().numpy(),
                )
                
                q_pred, td_loss, metric_loss = other_components
                
                if global_step % 100 == 0:
                    writer.add_scalar("losses/td_loss", jax.device_get(td_loss), global_step)
                    writer.add_scalar("losses/metric_loss", jax.device_get(metric_loss), global_step)
                    writer.add_scalar("losses/q_values", jax.device_get(q_pred).mean(), global_step)

            # update target network
            if global_step % args.target_network_frequency == 0:
                q_state = q_state.replace(
                    target_params=optax.incremental_update(q_state.params, q_state.target_params, args.tau)
                )

    ckpt = {'model': q_state, 'eval_retuns': evaluation_returns, 'train_returns': episodic_returns, 'lengths': episodic_lengths, 'args' : vars(args)}
    
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    save_args = orbax_utils.save_args_from_target(ckpt)

    orbax_dir = "runs/seed_" + str(args.seed) + '/orbax_ckpt/' + run_name    
    orbax_checkpointer.save(orbax_dir, ckpt, save_args = save_args)

    env.close()
    writer.close()
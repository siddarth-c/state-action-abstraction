import jax
import jax.numpy as jnp


def absolute_reward_diff(r1, r2):
	return jnp.abs(r1 - r2)

def l1_norm(x, y):
	return jnp.sum(x - y)

def l2_norm(x, y):
	return jnp.sqrt(jnp.sum(jnp.square(x - y)))

def huber_loss(targets, predictions, delta=1.0):
    x = jnp.abs(targets - predictions)
    return jnp.where(x <= delta, 0.5 * x**2, 0.5 * delta**2 + delta * (x - delta))

def squarify(x):
	batch_size = x.shape[0]
	if len(x.shape) > 1:
		representation_dim = x.shape[-1]
		return jnp.reshape(jnp.tile(x, batch_size), (batch_size, batch_size, representation_dim))
	return jnp.reshape(jnp.tile(x, batch_size), (batch_size, batch_size))


def representation_distances(first_representations, second_representations, distance_fn):

	batch_size = first_representations.shape[0]
	representation_dim = first_representations.shape[-1]
	first_squared_reps = squarify(first_representations)
	first_squared_reps = jnp.reshape(first_squared_reps, [batch_size**2, representation_dim])
	second_squared_reps = squarify(second_representations)
	second_squared_reps = jnp.transpose(second_squared_reps, axes=[1, 0, 2])
	second_squared_reps = jnp.reshape(second_squared_reps, [batch_size**2, representation_dim])
	distances = jax.vmap(distance_fn, in_axes=(0, 0))(first_squared_reps, second_squared_reps)

	return distances


def target_distances(representations, rewards, distance_fn, cumulative_gamma, metric, q_vals):
	"""Target distance using the metric operator."""
	
	# representations - (BATCH_SIZE, no_actions, MODULE_SIZE)
	# q_vals - (BATCH_SIZE, no_actions)
	
	batch_size, no_actions, module_size = representations.shape
	
	
	if metric == 1:
		# ! Selecting actions corresponding to the best Q-value
		actions_prime = jnp.expand_dims(jnp.argmax(q_vals, -1), -1) # (BATCH_SIZE, 1)
		# actions_prime has to be reshaped into (BATCH_SIZE, 1, MODULE_SIZE)
		# Repr_s_prime_a_prime - (BATCH_SIZE, MODULE_SIZE)
		Repr_s_prime_a_prime = jnp.take_along_axis(representations, jnp.expand_dims(actions_prime, -1).repeat(module_size, -1), 1).squeeze(1)
		next_state_similarities = representation_distances(Repr_s_prime_a_prime, Repr_s_prime_a_prime, distance_fn)
	
	if metric == 2:
		# ! Selecting actions corresponding to the max difference in Q-values
		# squarify(q_vals) - (BATCH_SIZE, BATCH_SIZE, no_actions)
		First_Q_s_prime_all_a_prime = squarify(q_vals).reshape(batch_size * batch_size, no_actions)
		Second_Q_s_prime_all_a_prime = squarify(q_vals).transpose(1, 0, 2).reshape(batch_size * batch_size, no_actions)
		actions_prime = jnp.expand_dims(jnp.argmax(jnp.abs(First_Q_s_prime_all_a_prime - Second_Q_s_prime_all_a_prime), 1), 1)
		# actions_prime - (BATCH_SIZE ^ 2, 1)
		next_state_similarities = []
		for a in range(no_actions):
			next_state_similarities.append(representation_distances(representations[:, a], representations[:, a], distance_fn))
		next_state_similarities = jnp.stack(next_state_similarities) # (no_actions, BATCH_SIZE ^ 2)
		next_state_similarities = next_state_similarities.transpose(1, 0) # (BATCH_SIZE ^ 2, no_actions)
		next_state_similarities = jnp.take_along_axis(next_state_similarities, actions_prime, axis = -1).squeeze(-1)
        
	if metric == 3: 
		# ! Selecting actions corresponding to the max difference in D_SA Metric (AAAI approach)
		next_state_similarities = []
		for a in range(no_actions):
			next_state_similarities.append(representation_distances(representations[:, a], representations[:, a], distance_fn))
		next_state_similarities = jnp.stack(next_state_similarities) # (no_actions, BATCH_SIZE ^ 2)
		next_state_similarities = next_state_similarities.transpose(1, 0) # (BATCH_SIZE ^ 2, no_actions)
		next_state_similarities = jnp.max(next_state_similarities, 1)
	
	squared_rews = squarify(rewards) # (BATCH_SIZE, BATCH_SIZE)
	squared_rews_transp = jnp.transpose(squared_rews) # Same but inverted
	squared_rews = squared_rews.reshape((squared_rews.shape[0]**2))
	squared_rews_transp = squared_rews_transp.reshape((squared_rews_transp.shape[0]**2))
	reward_diffs = absolute_reward_diff(squared_rews, squared_rews_transp)

	return jax.lax.stop_gradient(reward_diffs + cumulative_gamma * next_state_similarities)
import numpy as np


class RolloutWorker:
    def __init__(self, venv, policy, params, evaluate=False):
        """Rollout worker generates experience by interacting with one or many environments.

        Args:
            policy (object): the policy that is used to act
            dims (dict of ints): the dimensions for observations (o), goals (g), and actions (u)
            rollout_batch_size (int): the number of parallel rollouts that should be used
            noise_eps (float): scale of the additive Gaussian noise
            random_eps (float): probability of selecting a completely random action
        """
        self.venv = venv
        self.policy = policy
        self.dims = params['dims']
        self.T = params['T']
        self.rollout_batch_size = params['num_workers']
        self.clip_obs = params['clip_obs']
        self.evaluate = evaluate

        self.noise_eps = params['noise_eps'] if not evaluate else 0
        self.random_eps = params['random_eps'] if not evaluate else 0

        self.info_keys = [key.replace('info_', '') for key in params['dims'].keys() if key.startswith('info_')]

        self.reset_all_rollouts()
        self.counter = 0

    def reset_all_rollouts(self):
        self.obs_dict = self.venv.reset()
        self.initial_o = self.obs_dict['observation']
        self.initial_ag = self.obs_dict['achieved_goal']
        self.g = self.obs_dict['desired_goal'].astype(np.float32)

    def generate_rollouts(self):
        """Performs `rollout_batch_size` rollouts in parallel for time horizon `T` with the current
        policy acting on it accordingly.
        """
        self.reset_all_rollouts()

        # compute observations
        o = np.empty((self.rollout_batch_size, self.dims['o']), np.float32)   # observations
        ag = np.empty((self.rollout_batch_size, self.dims['g']), np.float32)  # achieved goals
        o[:] = self.initial_o
        ag[:] = self.initial_ag

        # generate episodes
        obs, achieved_goals, acts, goals, successes = [], [], [], [], []
        info_values = np.empty((self.T-1, self.rollout_batch_size, 1), np.float32)
        for t in range(self.T):
            actions = self.policy.act(
                o, self.g,
                noise_eps=self.noise_eps,
                random_eps=self.random_eps)

            # compute new states and observations
            actions = actions.cpu().detach().numpy()  # addition!! caution!!
            obs_dict_new, _, done, info = self.venv.step(actions)
            o_new = obs_dict_new['observation']
            ag_new = obs_dict_new['achieved_goal']
            success = np.array([i['is_success'] for i in info])

            # terminate rollouts whenever any env is done. Don't add obs from done envs
            if any(done): break

            for i, info_dict in enumerate(info):
                info_values[t, i] = info[i]['is_success']

            obs.append(o.copy())
            achieved_goals.append(ag.copy())
            successes.append(success.copy())
            acts.append(actions.copy())
            goals.append(self.g.copy())
            o[...] = o_new
            ag[...] = ag_new

        self.mean_success = np.mean(np.array(successes)[-1, :])  # success is only on the last timestep

        obs.append(o.copy())
        achieved_goals.append(ag.copy())

        episode = dict(o=obs,
                       u=acts,
                       g=goals,
                       ag=achieved_goals)

        episode['info_is_success'] = info_values

        episode_batch = {}
        for key in episode.keys():
            val = np.array(episode[key]).copy()
            # make inputs batch-major instead of time-major
            episode_batch[key] = val.swapaxes(0, 1)

        episode_batch['o'] = np.clip(episode_batch['o'], -self.clip_obs, self.clip_obs)
        return episode_batch

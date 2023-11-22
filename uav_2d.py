import numpy as np
import env as cpp_env

class ManyUavEnv:

    TARGET_R = 100
    COLLISION_R = 30

    def __init__(self, uav_cnt, seed):
        self._cpp_env = cpp_env.ManyUavEnv(uav_cnt, seed)
        self._viewer = None

    def reset(self):
        self._cpp_env.reset()
        obs = self._cpp_env.getObservations()
        return np.array(obs)

    def step(self, actions):
        self._cpp_env.step(actions)
        obs = np.array(self._cpp_env.getObservations())
        rewards = np.array(self._cpp_env.getRewards())
        done = self._cpp_env.isDone()
        return obs, rewards, done, {}

class EnvWrapper:#需要包装的

    def __init__(self, n_agents):
        self._env = ManyUavEnv(n_agents, 123)
        self._n_agents = n_agents
        self._global_center = True

    def set_global_center(self, value):
        self._global_center = value

    def reset(self):
        s = self._env.reset()

        result = {}
        for i in range(self._n_agents):
            result[f'uav_{i}'] = np.copy(s[i])
        return result

    def step(self, actions):
        act = []
        for i in range(self._n_agents):
            act.append(actions[f'uav_{i}'] * np.array([np.pi / 4, 1.0]))
        s, r, done, info = self._env.step(act)

        result_s = {}
        result_r = {}
        result_d = {}
        for i in range(self._n_agents):
            result_s[f'uav_{i}'] = np.copy(s[i])
            result_r[f'uav_{i}'] = r[i]
            result_d[f'uav_{i}'] = done

        result_d['__all__'] = all(result_d.values())
        return result_s, result_r, result_d, {}


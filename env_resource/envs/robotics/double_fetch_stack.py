import os
import tempfile
import numpy as np
import torch
from gym import utils
from gym.envs.robotics import fetch_env, rotations
from env_resource.envs.robotics.robot_utils import ctrl_set_action, mocap_set_action
import gym.envs.robotics.utils as robotics_utils
from env_resource.envs.assets.fetch.generate_xml_stack import generate_xml
from env_resource.envs.assets.fetch.multi_generate_xml_stack import generate_multi_xml
from gym import spaces

MODEL_XML_PATH = os.path.join(os.path.dirname(
    __file__), '../assets', 'fetch', 'pick_and_place_stack.xml')
MODEL_XML_PATH3 = os.path.join(os.path.dirname(__file__), '../assets', 'fetch', 'pick_and_place_stack3.xml')


class DoubleFetchStackEnv(fetch_env.FetchEnv, utils.EzPickle):
    def __init__(self, reward_type='sparse', random_gripper=True, random_box=True, random_ratio=1.0, n_object=2,
                 n_obs=1):
        initial_qpos = {
            'robot0:slide0': 0.0,
            'robot0:slide1': 0.0,
            'robot0:slide2': 0.0,

            'robot1:slide0': 0.0,
            'robot1:slide1': 0.0,
            'robot1:slide2': 0.0,
        }
        for i in range(n_object):
            initial_qpos['object%d:joint' % i] = [
                0.95 + 0.05 * i, 0.75 + 0.1 * (i - n_object // 2), 0.41, 1., 0., 0., 0.]
        self.random_gripper = random_gripper
        self.random_box = random_box
        self.random_ratio = random_ratio
        self.n_object = n_object
        self.task_array = []
        for i in range(self.n_object):
            self.task_array.extend([(i + 1, j) for j in range(i + 1)])
        with tempfile.NamedTemporaryFile(mode='wt',
                                         dir=os.path.join(os.path.dirname(
                                             __file__), '../assets', 'fetch'),
                                         delete=False, suffix=".xml") as fp:
            fp.write(generate_xml(self.n_object))
            model_path = fp.name
        self.task_mode = 0  # 0: pick and place, 1: stack
        fetch_env.FetchEnv.__init__(
            self, model_path, has_object=True, block_gripper=False, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=True, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type=reward_type)
        os.remove(model_path)
        utils.EzPickle.__init__(self)
        self.size_object = self.sim.model.geom_size[self.sim.model.geom_name2id(
            'object0')]
        self.size_obstacle = np.array([0.15, 0.15, 0.15])
        self.action_space = spaces.Box(-1., 1., shape=(8,), dtype='float32')
        self.initial_gripper_xpos_0 = self.sim.data.get_site_xpos('robot0:grip').copy()
        self.initial_gripper_xpos_1 = self.sim.data.get_site_xpos('robot1:grip').copy()
        self.initial_gripper_xpos = (self.initial_gripper_xpos_0 + self.initial_gripper_xpos_1) / 2
        self.n_obs = n_obs

    def _get_obs(self):
        grip_pos_0 = self.sim.data.get_site_xpos('robot0:grip')
        grip_pos_1 = self.sim.data.get_site_xpos('robot1:grip')
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp_0 = self.sim.data.get_site_xvelp('robot0:grip') * dt
        grip_velp_1 = self.sim.data.get_site_xvelp('robot1:grip') * dt
        robot_qpos, robot_qvel = robotics_utils.robot_get_obs(self.sim)
        if self.has_object:
            object_pos = [self.sim.data.get_site_xpos('object' + str(i)) if i in self.selected_objects else np.zeros(3)
                          for i in range(self.n_object)]
            # object_pos = [self.sim.data.get_site_xpos('object' + str(i)) for i in range(self.current_nobject)] \
            #              + [np.zeros(3) for _ in range(self.current_nobject, self.n_object)]
            # rotations
            object_rot = [rotations.mat2euler(self.sim.data.get_site_xmat('object' + str(i)))
                          if i in self.selected_objects else np.zeros(3) for i in range(self.n_object)]
            # object_rot = [rotations.mat2euler(self.sim.data.get_site_xmat('object' + str(i))) for i in range(self.current_nobject)] \
            #              + [np.zeros(3) for _ in range(self.current_nobject, self.n_object)]
            # velocities
            object_velp = [self.sim.data.get_site_xvelp('object' + str(i)) * dt
                           if i in self.selected_objects else np.zeros(3) for i in range(self.n_object)]
            # object_velp = [self.sim.data.get_site_xvelp('object' + str(i)) * dt for i in range(self.current_nobject)] \
            #               + [np.zeros(3) for _ in range(self.current_nobject, self.n_object)]
            object_velr = [self.sim.data.get_site_xvelr('object' + str(i)) * dt
                           if i in self.selected_objects else np.zeros(3) for i in range(self.n_object)]
            # object_velr = [self.sim.data.get_site_xvelr('object' + str(i)) * dt for i in range(self.current_nobject)] \
            #               + [np.zeros(3) for _ in range(self.current_nobject, self.n_object)]
            # gripper state
            # object_rel_pos = [pos - grip_pos for pos in object_pos]
            object_rel_pos_0 = [object_pos[i] - grip_pos_0 if i in self.selected_objects else np.zeros(3) for i in
                                range(self.n_object)]
            object_rel_pos_1 = [object_pos[i] - grip_pos_1 if i in self.selected_objects else np.zeros(3) for i in
                                range(self.n_object)]
            # object_rel_pos = [object_pos[i] - grip_pos for i in range(self.current_nobject)] \
            #                  + [np.zeros(3) for _ in range(self.current_nobject, self.n_object)]
            # object_velp = [velp - grip_velp for velp in object_velp]
            object_velp_0 = [object_velp[i] - grip_velp_0 if i in self.selected_objects else np.zeros(3) for i in
                             range(self.n_object)]
            object_velp_1 = [object_velp[i] - grip_velp_1 if i in self.selected_objects else np.zeros(3) for i in
                             range(self.n_object)]
            # object_velp = [object_velp[i] - grip_velp for i in range(self.current_nobject)] \
            #               + [np.zeros(3) for _ in range(self.current_nobject, self.n_object)]

            object_pos = np.concatenate(object_pos)
            object_rot = np.concatenate(object_rot)
            object_velp_0 = np.concatenate(object_velp_0)
            object_velp_1 = np.concatenate(object_velp_1)
            object_velr = np.concatenate(object_velr)
            object_rel_pos_0 = np.concatenate(object_rel_pos_0)
            object_rel_pos_1 = np.concatenate(object_rel_pos_1)

        else:
            object_pos = object_rot = object_velp_0 = object_velp_1 = object_velr = object_rel_pos_0 \
                = object_rel_pos_1 = np.zeros(0)
        gripper_state_0 = robot_qpos[13:15]
        gripper_state_1 = robot_qpos[-2:]
        # change to a scalar if the gripper is made symmetric
        gripper_vel_0 = robot_qvel[13:15] * dt
        gripper_vel_1 = robot_qvel[-2:] * dt

        if not self.has_object:
            achieved_goal = np.concatenate([grip_pos_0.copy(), grip_pos_1.copy()])
            grip_dist_0 = grip_dist_1 = None
        else:
            one_hot = self.goal[3:]
            idx = np.argmax(one_hot)
            achieved_goal = np.concatenate(
                [object_pos[3 * idx: 3 * (idx + 1)], one_hot])
            grip_dist_0 = object_rel_pos_0[3 * idx: 3 * (idx + 1)]
            grip_dist_1 = object_rel_pos_1[3 * idx: 3 * (idx + 1)]
            grip_dist_0 = np.linalg.norm(grip_dist_0)
            grip_dist_1 = np.linalg.norm(grip_dist_1)

        task_one_hot = np.zeros(2)
        task_one_hot[self.task_mode] = 1
        obs = np.concatenate([
            object_pos.ravel(), object_rot.ravel(),
            object_velr.ravel(),
            grip_pos_0, object_rel_pos_0.ravel(
            ), gripper_state_0,
            object_velp_0.ravel(), grip_velp_0, gripper_vel_0, grip_pos_1, object_rel_pos_1.ravel(
            ), gripper_state_1,
            object_velp_1.ravel(), grip_velp_1, gripper_vel_1, task_one_hot
        ])  # dim 60
        obs_0 = np.concatenate([
            object_pos.ravel(), object_rot.ravel(),
            object_velr.ravel(),
            grip_pos_0, object_rel_pos_0.ravel(
            ), gripper_state_0,
            object_velp_0.ravel(), grip_velp_0, gripper_vel_0, grip_pos_1, gripper_state_1, grip_velp_1, gripper_vel_1,
            task_one_hot
        ])  # dim 37
        # print("object_rel_pos_0.ravel():")
        # print(object_rel_pos_0.ravel().shape)
        # print(object_rel_pos_0.ravel())
        # print(obs_0[12:15])
        # print(np.linalg.norm(obs_0[12:15]))

        obs_1 = np.concatenate([
            object_pos.ravel(), object_rot.ravel(),
            object_velr.ravel(),
            grip_pos_1, object_rel_pos_1.ravel(
            ), gripper_state_1,
            object_velp_1.ravel(), grip_velp_1, gripper_vel_1, grip_pos_0, gripper_state_0, grip_velp_0, gripper_vel_0,
            task_one_hot
        ])
        # print("object_rel_pos_1.ravel():")
        # print(object_rel_pos_1.ravel().shape)
        # print(object_rel_pos_1.ravel())
        # print(obs_1[12:15])
        return {
            'observation': obs.copy(),
            'obs_1': obs_0.copy(),
            'obs_2': obs_1.copy(),
            # 'achieved_goal': achieved_goal.copy(),
            'achieved_goal': achieved_goal.copy(),
            # 'desired_goal': self.goal.copy(),
            'grip_dist_0': grip_dist_0,
            'grip_dist_1': grip_dist_1,
            'desired_goal': self.goal.copy(),
        }

    def get_obs(self):
        # print('in get_obs, goal', self.goal)
        return self._get_obs()

    def set_goal(self, goal):
        self.goal = goal.copy()

    def switch_obs_goal(self, obs, goal, task):
        obs = obs.copy()
        if isinstance(obs, dict):
            goal_idx = np.argmax(goal[3:])
            obs['achieved_goal'] = np.concatenate(
                [obs['observation'][3 + 3 * goal_idx: 3 + 3 * (goal_idx + 1)], goal[3:]])
            obs['desired_goal'] = goal
            assert task is not None
            obs['observation'][-2:] = 0
            obs['observation'][-2 + task] = 1
        elif isinstance(obs, np.ndarray):
            goal_idx = np.argmax(goal[3:])
            obs_dim = self.observation_space['observation'].shape[0]
            goal_dim = self.observation_space['achieved_goal'].shape[0]
            obs[obs_dim:obs_dim + 3] = obs[3 +
                                           goal_idx * 3: 3 + (goal_idx + 1) * 3]
            obs[obs_dim + 3:obs_dim + goal_dim] = goal[3:]
            obs[obs_dim + goal_dim:obs_dim + goal_dim * 2] = goal[:]
            assert task is not None
            obs[obs_dim - 2:obs_dim] = 0
            obs[obs_dim - 2 + task] = 1
        else:
            raise TypeError
        return obs

    def get_state(self):
        return self.sim.get_state()

    def set_state(self, state):
        self.sim.set_state(state)
        self.sim.forward()

    def set_current_nobject(self, current_nobject):
        self.current_nobject = current_nobject

    def set_selected_objects(self, selected_objects):
        self.selected_objects = selected_objects

    def set_task_mode(self, task_mode):
        self.task_mode = int(task_mode)

    def set_random_ratio(self, random_ratio):
        self.random_ratio = random_ratio

    def set_task_array(self, task_array):
        self.task_array = task_array.copy()

    def _reset_sim(self):
        self.sim.set_state(self.initial_state)
        if self.random_gripper:
            mocap_pos_0 = np.concatenate([self.np_random.uniform([1.05, 0.6], [1.3, 0.9]), [0.5]])
            mocap_pos_1 = np.concatenate([self.np_random.uniform([1.3, 0.6], [1.55, 0.9]), [0.5]])
            self.sim.data.set_mocap_pos('robot0:mocap', mocap_pos_0)
            self.sim.data.set_mocap_pos('robot1:mocap', mocap_pos_1)

            for _ in range(10):
                self.sim.step()
            self._step_callback()

        def is_valid(objects_xpos):
            for id1 in range(len(objects_xpos)):
                for id2 in range(id1 + 1, len(objects_xpos)):
                    if abs(objects_xpos[id1][0] - objects_xpos[id2][0]) < 2 * self.size_object[0] and \
                            abs(objects_xpos[id1][1] - objects_xpos[id2][1]) < 2 * self.size_object[1]:
                        return False
            return True

        # Set task.
        if self.np_random.uniform() < self.random_ratio:
            self.task_mode = 0  # pick and place
        else:
            self.task_mode = 1
        # Randomize start position of object.
        if self.has_object:
            # self.current_nobject = np.random.randint(0, self.n_object) + 1
            # self.sample_easy = (self.task_mode == 1)
            self.has_base = False
            task_rand = self.np_random.uniform()
            # if self.n_object == 2:
            #     task_array = [(1, 0), (2, 0), (2, 1)]
            #     self.current_nobject, base_nobject = task_array[int(task_rand * len(task_array))]
            # else:
            #     task_array = [(1, 0), (2, 0), (2, 1), (3, 0), (3, 1), (3, 2)]
            #     self.current_nobject, base_nobject = task_array[int(task_rand * len(task_array))]
            self.current_nobject, base_nobject = self.task_array[int(
                task_rand * len(self.task_array))]
            self._base_nobject = base_nobject
            self.selected_objects = np.random.choice(
                np.arange(self.n_object), self.current_nobject, replace=False)
            self.tower_height = self.height_offset + \
                                (base_nobject - 1) * self.size_object[2] * 2
            # if self.random_box and self.np_random.uniform() < self.random_ratio:
            if self.random_box:
                if base_nobject > 0:
                    self.has_base = True
                    objects_xpos = []
                    # base_nobject = np.random.randint(1, self.current_nobject)
                    self.maybe_goal_xy = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.obj_range,
                                                                                                self.obj_range, size=2)
                    for i in range(base_nobject):
                        objects_xpos.append(np.concatenate(
                            [self.maybe_goal_xy.copy(), [self.height_offset + i * 2 * self.size_object[2]]]))
                    for i in range(base_nobject, self.current_nobject):
                        objects_xpos.append(np.concatenate([self.initial_gripper_xpos[:2] + self.np_random.uniform(
                            -self.obj_range, self.obj_range, size=2), [self.height_offset]]))
                    while not is_valid(objects_xpos[base_nobject - 1:]):
                        for i in range(base_nobject, self.current_nobject):
                            objects_xpos[i][:2] = self.initial_gripper_xpos[:2] + self.np_random.uniform(
                                -self.obj_range, self.obj_range, size=2)
                    import random
                    random.shuffle(objects_xpos)
                else:
                    objects_xpos = []
                    for i in range(self.current_nobject):
                        objects_xpos.append(np.concatenate([self.initial_gripper_xpos[:2] + self.np_random.uniform(
                            -self.obj_range, self.obj_range, size=2), [self.height_offset]]))
                    while not is_valid(objects_xpos):
                        for i in range(self.current_nobject):
                            objects_xpos[i][:2] = self.initial_gripper_xpos[:2] + self.np_random.uniform(
                                -self.obj_range, self.obj_range, size=2)
            else:
                raise NotImplementedError

            # Set the position of obstacle. (free joint)
            for i in range(self.n_object):
                object_qpos = self.sim.data.get_joint_qpos(
                    'object%d:joint' % i)
                # if i < self.current_nobject:
                #     object_qpos[:3] = objects_xpos[i]
                #     # object_qpos[2] = self.height_offset
                if i in self.selected_objects:
                    object_qpos[:3] = objects_xpos[np.where(
                        self.selected_objects == i)[0][0]]
                else:
                    # This is coordinate in physical simulator, not the observation
                    object_qpos[:3] = np.array([-1 - i, -1, 0])
                self.sim.data.set_joint_qpos('object%d:joint' % i, object_qpos)

        self.sim.forward()
        return True

    def _sample_goal(self):
        if not hasattr(self, 'size_object'):
            self.size_object = self.sim.model.geom_size[self.sim.model.geom_name2id('object0')]
        if not hasattr(self, 'current_nobject'):
            self.current_nobject = self.n_object
        if not hasattr(self, 'selected_objects'):
            self.selected_objects = np.arange(self.current_nobject)
        # if not hasattr(self, 'sample_easy'):
        #     self.sample_easy = False

        if self.task_mode == 1:
            if self.has_base:
                # base_nobject > 0
                # Randomize goal height here.
                goal_height = self.height_offset + np.random.randint(self._base_nobject, self.current_nobject) * 2 * \
                              self.size_object[2]
                goal = np.concatenate([self.maybe_goal_xy, [goal_height]])
                # g_idx = np.random.randint(self.current_nobject)
                g_idx = np.random.choice(self.selected_objects)
                _count = 0
                while abs(self.sim.data.get_joint_qpos('object%d:joint' % g_idx)[0] - self.maybe_goal_xy[0]) < \
                        self.size_object[0] \
                        and abs(self.sim.data.get_joint_qpos('object%d:joint' % g_idx)[1] - self.maybe_goal_xy[1]) < \
                        self.size_object[1]:
                    _count += 1
                    if _count > 100:
                        print(self.maybe_goal_xy, self.sim.data.get_joint_qpos('object0:joint'),
                              self.sim.data.get_joint_qpos('object1:joint'))
                        print(self.current_nobject, self.n_object)
                        raise RuntimeError
                    # g_idx = np.random.randint(self.current_nobject)
                    g_idx = np.random.choice(self.selected_objects)
            else:
                assert self._base_nobject == 0
                goal_height = self.height_offset + np.random.randint(0, self.current_nobject) * 2 * self.size_object[2]
                goal = np.concatenate([self.initial_gripper_xpos[:2] + self.np_random.uniform(
                    -self.target_range, self.target_range, size=2), [goal_height]])
                # g_idx = np.random.randint(self.current_nobject)
                g_idx = np.random.choice(self.selected_objects)
            one_hot = np.zeros(self.n_object)
            one_hot[g_idx] = 1
        else:
            # Pick and place
            # g_idx = np.random.randint(self.current_nobject)
            g_idx = np.random.choice(self.selected_objects)
            if hasattr(self, 'has_base') and self.has_base:
                while abs(self.sim.data.get_joint_qpos('object%d:joint' % g_idx)[0] - self.maybe_goal_xy[0]) < 0.01 \
                        and abs(
                    self.sim.data.get_joint_qpos('object%d:joint' % g_idx)[1] - self.maybe_goal_xy[1]) < 0.01:
                    g_idx = np.random.choice(self.selected_objects)
            one_hot = np.zeros(self.n_object)
            one_hot[g_idx] = 1
            goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(-self.target_range, self.target_range, size=3)
            goal[2] = self.height_offset
            if self.np_random.uniform() < 0.5:
                goal[2] += self.np_random.uniform(0, 0.25)
        goal = np.concatenate([goal, one_hot])
        return goal.copy()

    def _is_success(self, achieved_goal, desired_goal):
        raise NotImplementedError

    def _goal_distance(self, achieved_goal, goal):
        # print(achieved_goal.shape)
        # print(goal.shape)
        assert achieved_goal.shape == goal.shape
        return np.linalg.norm(achieved_goal - goal, axis=-1)

    def _goal_distance_tensor(self, achieved_goal, goal):
        # print(achieved_goal.shape)
        # print(goal.shape)
        assert achieved_goal.shape == goal.shape
        return torch.norm(achieved_goal - goal, dim=-1)

    def _is_stacked(self, achieved_goal, goal, other_objects_pos):
        assert len(other_objects_pos) == 3 * (self.n_object - 1)
        current_nobject = np.sum([np.linalg.norm(other_objects_pos[3 * i: 3 * (i + 1)]) > 1e-3
                                  for i in range(len(other_objects_pos) // 3)]) + 1
        stack = True
        n_stores = min([int(round((goal[2] - self.height_offset) /
                                  (2 * self.size_object[2]))), int(current_nobject - 1)])
        for h in range(n_stores):
            stack = False
            for i in range(self.n_object - 1):
                # iterate over other objects to see if one of them fills the position
                if abs(other_objects_pos[3 * i + 2] - (
                        achieved_goal[2] - 2 * self.size_object[2] * (h + 1))) < 0.01 \
                        and abs(other_objects_pos[3 * i] - achieved_goal[0]) < self.size_object[0] \
                        and abs(other_objects_pos[3 * i + 1] - achieved_goal[1]) < self.size_object[1]:
                    stack = True
                    break
            if not stack:
                break
        return stack

    def _is_stacked_tensor(self, achieved_goal, goal, other_objects_pos):
        assert len(other_objects_pos) == 3 * (self.n_object - 1)
        current_nobject = torch.tensor([torch.norm(other_objects_pos[3 * i: 3 * (i + 1)]) > 1e-3
                                        for i in range(len(other_objects_pos) // 3)],
                                       device=achieved_goal.device).sum() + 1
        stack = True
        n_stores = min([int(torch.round((goal[2] - self.height_offset) /
                                        (2 * self.size_object[2]))), current_nobject - 1])
        for h in range(n_stores):
            stack = False
            for i in range(self.n_object - 1):
                # iterate over other objects to see if one of them fills the position
                if abs(other_objects_pos[3 * i + 2] - (
                        achieved_goal[2] - 2 * self.size_object[2] * (h + 1))) < 0.01 \
                        and abs(other_objects_pos[3 * i] - achieved_goal[0]) < self.size_object[0] \
                        and abs(other_objects_pos[3 * i + 1] - achieved_goal[1]) < self.size_object[1]:
                    stack = True
                    break
            if not stack:
                break
        return stack

    def compute_reward(self, observation, goal, info, gpu=False):
        if not gpu:
            r, _ = self.compute_reward_and_success(observation, goal, info)
        else:
            r, _ = self.compute_reward_and_success_tensor(observation, goal, info)
        return r

    def compute_reward_and_success(self, observation, goal, info):
        obs_len = len(observation.shape)
        # observation_dim = self.observation_space['observation'].shape[-1]
        observation_dim = observation.shape[-1]
        goal_dim = self.observation_space['achieved_goal'].shape[-1]
        T = goal.shape[0]
        if obs_len == 2:
            task_mode = np.argmax(
                observation[:, observation_dim - 2: observation_dim])
            one_hot = goal[:, 3:]
            idx = np.argmax(one_hot, axis=-1)
            achieved_goal = np.array([observation[i, 3 * idx[i]: 3 * (idx[i] + 1)] for i in range(len(idx))])
        else:
            task_mode = np.argmax(
                observation[observation_dim - 2: observation_dim])
            one_hot = goal[3:]
            idx = np.argmax(one_hot)
            achieved_goal = observation[3 * idx: 3 * (idx + 1)]

        if isinstance(info, dict) and isinstance(info['previous_obs'], np.ndarray):
            info['previous_obs'] = dict(observation=info['previous_obs'][:observation_dim],
                                        achieved_goal=info['previous_obs'][observation_dim: observation_dim + goal_dim],
                                        desired_goal=info['previous_obs'][observation_dim + goal_dim:])
        if task_mode == 0:
            if obs_len == 2:
                if self.reward_type == 'dense':
                    r = -np.linalg.norm(achieved_goal - goal[:, :3], axis=-1)
                else:
                    # print(goal.shape)
                    r = -((self._goal_distance(achieved_goal,
                                               goal[:, :3]) > self.distance_threshold).astype(np.float32))
                    mask = abs(achieved_goal[:, 2] - goal[:, 2]) < 0.01
                    r = r * mask + mask - 1
                success = (np.linalg.norm(achieved_goal - goal[:, :3], axis=-1) < self.distance_threshold) * (abs(
                    achieved_goal[:, 2] - goal[:, 2]) < 0.01)
                if self.reward_type == 'dense':
                    r = 0.1 * r + success
            else:
                if self.reward_type == 'dense':
                    r = -np.linalg.norm(achieved_goal - goal[:3])
                else:
                    r = -((self._goal_distance(achieved_goal,
                                               goal[:3]) > self.distance_threshold).astype(np.float32))
                    if abs(achieved_goal[2] - goal[2]) > 0.01:
                        r = -1
                success = (np.linalg.norm(achieved_goal - goal[:3]) < self.distance_threshold) * (abs(
                    achieved_goal[2] - goal[2]) < 0.01)
                if self.reward_type == 'dense':
                    r = 0.1 * r + success
        else:
            if self.reward_type == 'dense':
                r_achieve = -np.linalg.norm(achieved_goal - goal[:3])
                if np.linalg.norm(achieved_goal - goal[:3]) < self.distance_threshold:
                    r = r_achieve
                    gripper_far = np.linalg.norm(
                        observation[:3] - achieved_goal) > self.distance_threshold
                    other_objects_pos = np.concatenate([observation[: 3 * idx],
                                                        observation[3 * (idx + 1): 3 * self.n_object]])
                    stack = self._is_stacked(
                        achieved_goal, goal, other_objects_pos)
                    success = gripper_far and stack
                else:
                    r = r_achieve
                    success = 0.0
                r = 0.1 * r + success
            elif self.reward_type == 'incremental':
                if self.current_nobject == 2:
                    achieved_goal = observation[: 3 * self.n_object]
                    desired_goal = np.concatenate([goal[:3], goal[:3]])
                    idx = np.argmin(one_hot)
                    # Assume only 2 objects
                    desired_goal[2 + 3 * idx] = self.height_offset
                elif self.current_nobject == 1:
                    achieved_goal = observation[3 * idx: 3 * (idx + 1)]
                    desired_goal = goal[:3]
                else:
                    raise NotImplementedError
                r_achieve = -np.sum([(self._goal_distance(achieved_goal[3 * i: 3 * (i + 1)], desired_goal[3 * i: 3 * (
                        i + 1)]) > self.distance_threshold).astype(np.float32) for i in
                                     range(self.current_nobject)])
                r_achieve = np.asarray(r_achieve)
                gripper_far = np.all(
                    [np.linalg.norm(observation[:3] - achieved_goal[3 * i: 3 * (i + 1)]) > 2 * self.distance_threshold
                     for i in range(self.current_nobject)])
                np.putmask(r_achieve, r_achieve == 0, gripper_far)
                r = r_achieve
                success = (r > 0.5)
            else:
                # print(achieved_goal.shape)
                # print(goal.shape)
                if obs_len == 2:
                    r_achieve = -(self._goal_distance(achieved_goal, goal[:, :3]) > self.distance_threshold).astype(
                        np.float32)
                    mask = (abs(achieved_goal[:, 2] - goal[:, 2]) - np.ones(T) * 0.01 <= 0).astype(np.float32)
                    r_achieve = r_achieve * mask + mask - 1
                    # r_achieve = -1 * np.ones(T)
                    mask = (r_achieve + np.ones(T) > 0).astype(np.float32)
                    r = r_achieve * mask + mask - 1
                    # Check if stacked
                    for i in range(T):
                        other_objects_pos = np.concatenate([observation[i][: 3 * idx[i]], observation[i][3 * (
                                idx[i] + 1): 3 * self.n_object]])
                        stack = self._is_stacked(
                            achieved_goal[i], goal[i], other_objects_pos)
                        gripper_far = np.linalg.norm(
                            observation[i][:3] - achieved_goal) > 2 * self.distance_threshold
                        # print('stack', stack, 'gripper_far', gripper_far)
                        if stack and gripper_far:
                            r[i] = 0.0
                        else:
                            r[i] = -1.0
                    success = r > -0.5
                else:
                    r_achieve = -(self._goal_distance(achieved_goal, goal[:3]) > self.distance_threshold).astype(
                        np.float32)
                    if abs(achieved_goal[2] - goal[2]) > 0.01:
                        r_achieve = -1

                    if r_achieve < -0.5:
                        r = -1.0
                    else:
                        # Check if stacked
                        other_objects_pos = np.concatenate([observation[: 3 * idx],
                                                            observation[3 * (idx + 1): 3 * self.n_object]])
                        # print('other_objects_pos', other_objects_pos)
                        # print('achieved_goal', achieved_goal)
                        stack = self._is_stacked(
                            achieved_goal, goal, other_objects_pos)
                        if self.n_obs == 1:
                            gripper_far = np.linalg.norm(
                                observation[
                                self.n_object * 9:self.n_object * 9 + 3] - achieved_goal) > 2 * self.distance_threshold and (
                                                  np.linalg.norm(observation[
                                                                 self.n_object * 15 + 10:self.n_object * 15 + 13] - achieved_goal) > 2 * self.distance_threshold)
                        elif self.n_obs == 2:
                            gripper_far = (np.linalg.norm(
                                observation[
                                self.n_object * 9:self.n_object * 9 + 3] - achieved_goal) > 2 * self.distance_threshold) and (
                                                  np.linalg.norm(observation[
                                                                 self.n_object * 15 + 10:self.n_object * 15 + 13] - achieved_goal) > 2 * self.distance_threshold)
                        else:
                            raise NotImplementedError

                        if stack and gripper_far:
                            r = 0.0
                        else:
                            r = -1.0
                    success = r > -0.5
        return r, success

    def compute_reward_and_success_tensor(self, observation, goal, info):
        obs_len = len(observation.shape)
        # observation_dim = self.observation_space['observation'].shape[-1]
        observation_dim = observation.shape[-1]
        goal_dim = self.observation_space['achieved_goal'].shape[-1]
        T = goal.shape[0]
        if obs_len == 2:
            task_mode = torch.argmax(
                observation[:, observation_dim - 2: observation_dim])
            one_hot = goal[:, 3:]
            idx = torch.argmax(one_hot, dim=-1)
            achieved_goal = torch.stack([observation[i, 3 * idx[i]: 3 * (idx[i] + 1)] for i in range(len(idx))])
        else:
            task_mode = torch.argmax(
                observation[observation_dim - 2: observation_dim])
            one_hot = goal[3:]
            idx = torch.argmax(one_hot)
            achieved_goal = observation[3 * idx: 3 * (idx + 1)]
        if isinstance(info, dict) and isinstance(info['previous_obs'], np.ndarray):
            info['previous_obs'] = dict(observation=info['previous_obs'][:observation_dim],
                                        achieved_goal=info['previous_obs'][observation_dim: observation_dim + goal_dim],
                                        desired_goal=info['previous_obs'][observation_dim + goal_dim:])
        if task_mode == 0:
            if obs_len == 2:
                if self.reward_type == 'dense':
                    r = -torch.norm(achieved_goal - goal[:, :3])
                else:
                    # print(goal.shape)
                    r = -(self._goal_distance_tensor(achieved_goal,
                                                     goal[:, :3]) > self.distance_threshold).float()
                    mask = abs(achieved_goal[:, 2] - goal[:, 2]) < 0.01
                    r = r * mask + mask - 1
                success = torch.norm(achieved_goal - goal[:, :3]) < self.distance_threshold and abs(
                    achieved_goal[:, 2] - goal[:, 2]) < 0.01
                if self.reward_type == 'dense':
                    r = 0.1 * r + success
            else:
                if self.reward_type == 'dense':
                    r = -torch.norm(achieved_goal - goal[:3])
                else:
                    r = -((self._goal_distance_tensor(achieved_goal,
                                                      goal[:3]) > self.distance_threshold)).float()
                    if abs(achieved_goal[2] - goal[2]) > 0.01:
                        r = -1
                success = torch.norm(achieved_goal - goal[:3]) < self.distance_threshold and abs(
                    achieved_goal[2] - goal[2]) < 0.01
                if self.reward_type == 'dense':
                    r = 0.1 * r + success
        else:
            if self.reward_type == 'dense':
                r_achieve = -torch.norm(achieved_goal - goal[:3])
                if torch.norm(achieved_goal - goal[:3]) < self.distance_threshold:
                    r = r_achieve
                    gripper_far = (torch.norm(
                        observation[:3] - achieved_goal) > self.distance_threshold).float()
                    other_objects_pos = torch.cat((observation[: 3 * idx],
                                                   observation[3 * (idx + 1): 3 * self.n_object]))
                    stack = self._is_stacked_tensor(
                        achieved_goal, goal, other_objects_pos)
                    success = gripper_far and stack
                else:
                    r = r_achieve
                    success = 0.0
                r = 0.1 * r + success
            elif self.reward_type == 'incremental':
                if self.current_nobject == 2:
                    achieved_goal = observation[: 3 * self.n_object]
                    desired_goal = torch.cat((goal[:3], goal[:3]))
                    idx = torch.argmin(one_hot)
                    # Assume only 2 objects
                    desired_goal[2 + 3 * idx] = self.height_offset
                elif self.current_nobject == 1:
                    achieved_goal = observation[3 * idx: 3 * (idx + 1)]
                    desired_goal = goal[:3]
                else:
                    raise NotImplementedError
                r_achieve = -torch.stack(
                    [(self._goal_distance_tensor(achieved_goal[3 * i: 3 * (i + 1)], desired_goal[3 * i: 3 * (
                            i + 1)]) > self.distance_threshold) for i in
                     range(self.current_nobject)]).sum()
                # r_achieve = np.asarray(r_achieve)
                gripper_far = torch.stack(
                    [torch.norm(observation[0:3] - achieved_goal[3 * i: 3 * (i + 1)]) > 2 * self.distance_threshold
                     for i in range(self.current_nobject)]).all()
                np.putmask(r_achieve, r_achieve == 0, gripper_far)
                r = r_achieve
                success = (r > 0.5)
            else:
                # print(achieved_goal.shape)
                # print(goal.shape)
                if obs_len == 2:
                    r_achieve = -(self._goal_distance_tensor(achieved_goal,
                                                             goal[:, :3]) > self.distance_threshold).float()
                    mask = (abs(achieved_goal[:, 2] - goal[:, 2]) - torch.ones((T),
                                                                               device=observation.device) * 0.01 <= 0)
                    r_achieve = r_achieve * mask + mask - 1
                    # r_achieve = -1 * np.ones(T)
                    mask = (r_achieve + torch.ones((T), device=observation.device) > 0)
                    r = r_achieve * mask + mask - 1
                    # Check if stacked
                    for i in range(T):
                        other_objects_pos = torch.cat((observation[i][: 3 * idx[i]], observation[i][3 * (
                                idx[i] + 1): 3 * self.n_object]))
                        stack = self._is_stacked_tensor(
                            achieved_goal[i], goal[i], other_objects_pos)
                        gripper_far = torch.norm(
                            observation[i][:3] - achieved_goal) > 2 * self.distance_threshold
                        # print('stack', stack, 'gripper_far', gripper_far)
                        if stack and gripper_far:
                            r[i] = 0.0
                        else:
                            r[i] = -1.0
                    success = r > -0.5
                else:
                    r_achieve = -(self._goal_distance_tensor(achieved_goal, goal[:3]) > self.distance_threshold)
                    if abs(achieved_goal[2] - goal[2]) > 0.01:
                        r_achieve = -1

                    if r_achieve < -0.5:
                        r = -1.0
                    else:
                        # Check if stacked
                        other_objects_pos = torch.cat((observation[: 3 * idx],
                                                       observation[3 * (idx + 1): 3 * self.n_object]))
                        # print('other_objects_pos', other_objects_pos)
                        # print('achieved_goal', achieved_goal)
                        stack = self._is_stacked_tensor(
                            achieved_goal, goal, other_objects_pos)
                        gripper_far = torch.norm(
                            observation[
                            self.n_object * 9:self.n_object * 9 + 3] - achieved_goal) > 2 * self.distance_threshold and (
                                              torch.norm(observation[
                                                         self.n_object * 15 + 10:self.n_object * 15 + 13] - achieved_goal) > 2 * self.distance_threshold)

                        if stack and gripper_far:
                            r = 0.0
                        else:
                            r = -1.0
                    success = r > -0.5
        return r, success

    def _set_action(self, action):
        assert action.shape == (8,)
        action = action.copy()  # ensure that we don't change the action outside of this scope
        pos_ctrl_0, gripper_ctrl_0 = action[:3], action[3]
        pos_ctrl_1, gripper_ctrl_1 = action[4:7], action[7]

        pos_ctrl_0 *= 0.05  # limit maximum change in position
        pos_ctrl_1 *= 0.05
        rot_ctrl_0 = np.array([1., 0., 1., 0.])  # fixed rotation of the end effector, expressed as a quaternion
        rot_ctrl_1 = np.array([1., 0., 1., 0.])
        gripper_ctrl_0 = np.array([gripper_ctrl_0, gripper_ctrl_0])
        gripper_ctrl_1 = np.array([gripper_ctrl_1, gripper_ctrl_1])

        assert gripper_ctrl_0.shape == (2,)
        assert gripper_ctrl_1.shape == (2,)

        if self.block_gripper:
            gripper_ctrl_0 = np.zeros_like(gripper_ctrl_0)
            gripper_ctrl_1 = np.zeros_like(gripper_ctrl_1)

        action = np.concatenate([pos_ctrl_0, rot_ctrl_0, pos_ctrl_1,
                                 rot_ctrl_1, gripper_ctrl_0, gripper_ctrl_1])

        # Apply action to simulation.
        ctrl_set_action(self.sim, action)
        mocap_set_action(self.sim, action)

    def step(self, action):
        # print(self.action_space.low)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._set_action(action)
        previous_obs = self._get_obs()
        info = {'previous_obs': previous_obs, }
        self.sim.step()
        self._step_callback()
        obs = self._get_obs()

        # Coarsely check x,y position and z height
        intower_idx = list(filter(lambda idx: np.linalg.norm(
            obs['observation'][3 * idx: 3 * idx + 2] - obs['desired_goal'][:2]) < 0.025
                                              and abs((obs['observation'][3 * idx + 2] - self.height_offset)
                                                      - 0.05 * round(
            (obs['observation'][3 * idx + 2] - self.height_offset) / 0.05)) < 0.01,
                                  np.arange(self.n_object)))
        # TODO: more refined tower height calculation: from bottom to top, check if stacked properly.
        # Also, if target block is not placed at the desired level, it should be considered as part of tower.
        self.tower_height = self.height_offset - self.size_object[2] * 2
        for i in range(0, self.current_nobject):
            found = False
            for prob_idx in intower_idx:
                if self.height_offset + 2 * self.size_object[2] * i < obs['desired_goal'][2] - 0.01:
                    # Should not be target object
                    if prob_idx != np.argmax(obs['desired_goal'][3:]) and \
                            abs(obs['observation'][3 * prob_idx + 2] - (
                                    self.height_offset + 2 * self.size_object[2] * i)) < 0.01:
                        self.tower_height += 2 * self.size_object[2]
                        found = True
                        break
                else:
                    # desired goal height now, no need to check if it is target_block
                    if abs(obs['observation'][3 * prob_idx + 2] - (
                            self.height_offset + 2 * self.size_object[2] * i)) < 0.01:
                        self.tower_height += 2 * self.size_object[2]
                        found = True
                        break

            if not found:
                break

        done = False
        if len(obs['observation'].shape) != 1:
            print("in 814")
            print(len(obs['observation'].shape))
        reward, is_success = self.compute_reward_and_success(obs['observation'], self.goal, info)
        info['is_success'] = is_success
        return obs, reward, done, info

    def _viewer_setup(self):
        body_id_0 = self.sim.model.body_name2id('robot0:gripper_link')

        lookat = self.sim.data.body_xpos[body_id_0]
        for idx, value in enumerate(lookat):
            self.viewer.cam.lookat[idx] = value
        self.viewer.cam.distance = 2.0
        self.viewer.cam.azimuth = 180.
        self.viewer.cam.elevation = -90.

    def _render_callback(self):
        # Visualize target.
        sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()
        site_id = self.sim.model.site_name2id('target0')
        g_idx = np.argmax(self.goal[3:])
        object_id = self.sim.model.site_name2id('object%d' % g_idx)
        self.sim.model.site_pos[site_id] = self.goal[:3] - sites_offset[0]
        self.sim.model.site_rgba[site_id] = np.concatenate([self.sim.model.site_rgba[object_id][:3], [0.5]])
        self.sim.forward()

    def _env_setup(self, initial_qpos):
        for name, value in initial_qpos.items():
            self.sim.data.set_joint_qpos(name, value)
        robotics_utils.reset_mocap_welds(self.sim)
        self.sim.forward()

        # Move end effector into position.
        gripper_target_0 = np.array([-0.498, 0.005, -0.431 + self.gripper_extra_height]) + self.sim.data.get_site_xpos(
            'robot0:grip')
        gripper_target_1 = np.array([-0.498, 0.005, -0.431 + self.gripper_extra_height]) + self.sim.data.get_site_xpos(
            'robot1:grip')
        gripper_rotation_0 = np.array([0., 0., 0., 0.])
        gripper_rotation_1 = np.array([-0.7, 0., 0., 1.])

        self.sim.data.set_mocap_pos('robot0:mocap', gripper_target_0)
        self.sim.data.set_mocap_quat('robot0:mocap', gripper_rotation_0)

        self.sim.data.set_mocap_pos('robot1:mocap', gripper_target_1)
        self.sim.data.set_mocap_quat('robot1:mocap', gripper_rotation_1)
        for _ in range(10):
            self.sim.step()

        # Extract information for sampling goals.
        self.initial_gripper_xpos_0 = self.sim.data.get_site_xpos('robot0:grip').copy()
        self.initial_gripper_xpos_1 = self.sim.data.get_site_xpos('robot1:grip').copy()
        self.initial_gripper_xpos = (self.initial_gripper_xpos_0 + self.initial_gripper_xpos_1) / 2
        if self.has_object:
            self.height_offset = 0.42

    def _step_callback(self):
        if self.block_gripper:
            self.sim.data.set_joint_qpos('robot0:l_gripper_finger_joint', 0.)
            self.sim.data.set_joint_qpos('robot0:r_gripper_finger_joint', 0.)
            self.sim.data.set_joint_qpos('robot1:l_gripper_finger_joint', 0.)
            self.sim.data.set_joint_qpos('robot1:r_gripper_finger_joint', 0.)
            self.sim.forward()


class DoubleFetchStackEnv_v2(DoubleFetchStackEnv):
    '''
    Modified from FetchStackEnv, but with all blocks initialized on the table,
    goals for stacking sampled from every possible heights
    '''

    def __init__(self, reward_type='sparse', random_gripper=True, random_box=True, random_ratio=1.0, n_object=2):
        initial_qpos = {
            'robot0:slide0': 0.0,
            'robot0:slide1': 0.0,
            'robot0:slide2': 0.0,

            'robot1:slide0': 0.0,
            'robot1:slide1': 0.0,
            'robot1:slide2': 0.0,
        }
        for i in range(n_object):
            initial_qpos['object%d:joint' % i] = [1.25 + 0.05 * i, 0.53, 0.4, 1., 0., 0., 0.]
        self.random_gripper = random_gripper
        self.random_box = random_box
        self.random_ratio = random_ratio
        self.n_object = n_object
        self.task_array = []
        for i in range(self.n_object):
            # Every object should be initialized from table.
            self.task_array.extend([(i + 1, j) for j in range(0, min(2, i + 1))])
        with tempfile.NamedTemporaryFile(mode='wt',
                                         dir=os.path.join(os.path.dirname(__file__), '../assets', 'fetch'),
                                         # dir=os.path.join(os.path.dirname(__file__), '../assets', 'fetch', 'pick_and_place_stack3.xml'),
                                         delete=False, suffix=".xml") as fp:
            fp.write(generate_xml(self.n_object))
            model_path = fp.name
            # model_path = MODEL_XML_PATH3
        self.task_mode = 0  # 0: pick and place, 1: stack
        fetch_env.FetchEnv.__init__(
            self, model_path, has_object=True, block_gripper=False, n_substeps=20,  # originally block_gripper = False
            gripper_extra_height=0.2, target_in_the_air=True, target_offset=0.0,  # originally target_in_the_air = True
            obj_range=0.15, target_range=0.25, distance_threshold=0.05,  # originally target_range=0.15
            initial_qpos=initial_qpos, reward_type=reward_type)
        os.remove(model_path)
        utils.EzPickle.__init__(self)
        self.size_object = self.sim.model.geom_size[self.sim.model.geom_name2id('object0')]
        self.size_obstacle = np.array([0.15, 0.15, 0.15])
        self.action_space = spaces.Box(-1., 1., shape=(8,), dtype='float32')
        self.initial_gripper_xpos_0 = self.sim.data.get_site_xpos('robot0:grip').copy()
        self.initial_gripper_xpos_1 = self.sim.data.get_site_xpos('robot1:grip').copy()
        self.initial_gripper_xpos = (self.initial_gripper_xpos_0 + self.initial_gripper_xpos_1) / 2

    def _sample_goal(self):
        if not hasattr(self, 'size_object'):
            self.size_object = self.sim.model.geom_size[self.sim.model.geom_name2id('object0')]
        if not hasattr(self, 'current_nobject'):
            self.current_nobject = self.n_object
        if not hasattr(self, 'selected_objects'):
            self.selected_objects = np.arange(self.current_nobject)
        # if not hasattr(self, 'sample_easy'):
        #     self.sample_easy = False

        if self.task_mode == 1:
            if self.has_base:
                # base_nobject > 0
                # Randomize goal height here.
                goal_height = self.height_offset + np.random.randint(self._base_nobject, self.current_nobject) * 2 * \
                              self.size_object[2]
                goal = np.concatenate([self.maybe_goal_xy, [goal_height]])
                # g_idx = np.random.randint(self.current_nobject)
                g_idx = np.random.choice(self.selected_objects)
                _count = 0
                while abs(self.sim.data.get_joint_qpos('object%d:joint' % g_idx)[0] - self.maybe_goal_xy[0]) < \
                        self.size_object[0] \
                        and abs(self.sim.data.get_joint_qpos('object%d:joint' % g_idx)[1] - self.maybe_goal_xy[1]) < \
                        self.size_object[1]:
                    _count += 1
                    if _count > 100:
                        print(self.maybe_goal_xy, self.sim.data.get_joint_qpos('object0:joint'),
                              self.sim.data.get_joint_qpos('object1:joint'))
                        print(self.current_nobject, self.n_object)
                        raise RuntimeError
                    # g_idx = np.random.randint(self.current_nobject)
                    g_idx = np.random.choice(self.selected_objects)
            else:
                assert self._base_nobject == 0
                goal_height = self.height_offset + np.random.randint(0, self.current_nobject) * 2 * self.size_object[2]
                goal = np.concatenate([self.initial_gripper_xpos[:2] + self.np_random.uniform(
                    -self.target_range, self.target_range, size=2), [goal_height]])
                # g_idx = np.random.randint(self.current_nobject)
                g_idx = np.random.choice(self.selected_objects)
            one_hot = np.zeros(self.n_object)
            one_hot[g_idx] = 1
        else:
            # Pick and place
            # g_idx = np.random.randint(self.current_nobject)
            g_idx = np.random.choice(self.selected_objects)
            if hasattr(self, 'has_base') and self.has_base:
                while abs(self.sim.data.get_joint_qpos('object%d:joint' % g_idx)[0] - self.maybe_goal_xy[0]) < 0.01 \
                        and abs(
                    self.sim.data.get_joint_qpos('object%d:joint' % g_idx)[1] - self.maybe_goal_xy[1]) < 0.01:
                    g_idx = np.random.choice(self.selected_objects)
            one_hot = np.zeros(self.n_object)
            one_hot[g_idx] = 1
            goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(-self.target_range, self.target_range, size=3)
            goal[2] = self.height_offset
            if self.np_random.uniform() < 0.5:
                goal[2] += self.np_random.uniform(0, 0.45)
        goal = np.concatenate([goal, one_hot])
        return goal.copy()


class DoubleFetchStackEnv_v1(DoubleFetchStackEnv):
    '''
    Modified from FetchStackEnv, but with all blocks initialized on the table,
    goals for stacking sampled from every possible heights
    '''

    def __init__(self, reward_type='sparse', random_gripper=True, random_box=True, random_ratio=1.0, n_object=2,
                 n_robot=2, _all=False):
        initial_qpos = {
            'robot0:slide0': 0.0,
            'robot0:slide1': 0.0,
            'robot0:slide2': 0.0,

            'robot1:slide0': 0.0,
            'robot1:slide1': 0.0,
            'robot1:slide2': 0.0,
        }
        for i in range(n_object):
            initial_qpos['object%d:joint' % i] = [1.25 + 0.05 * i, 0.53, 0.4, 1., 0., 0., 0.]
        self.random_gripper = random_gripper
        self.random_box = random_box
        self.random_ratio = random_ratio
        self.n_object = n_object
        self.n_robot = n_robot
        self.task_array = []
        self._all = _all
        for i in range(self.n_object):
            # Every object should be initialized from table.
            self.task_array.extend([(i + 1, j) for j in range(0, min(2, i + 1))])
        self.task_robot_array = []
        for i in range(self.n_robot):
            self.task_robot_array.extend([(i + 1, j) for j in range(i + 1)])
        with tempfile.NamedTemporaryFile(mode='wt',
                                         dir=os.path.join(os.path.dirname(__file__), '../assets', 'fetch'),
                                         # dir=os.path.join(os.path.dirname(__file__), '../assets', 'fetch', 'pick_and_place_stack3.xml'),
                                         delete=False, suffix=".xml") as fp:
            fp.write(generate_xml(self.n_object))
            model_path = fp.name
            # model_path = MODEL_XML_PATH3
        self.task_mode = 0  # 0: pick and place, 1: stack
        fetch_env.FetchEnv.__init__(
            self, model_path, has_object=True, block_gripper=False, n_substeps=20,  # originally block_gripper = False
            gripper_extra_height=0.2, target_in_the_air=True, target_offset=0.0,  # originally target_in_the_air = True
            obj_range=0.15, target_range=0.25, distance_threshold=0.05,  # originally target_range=0.15
            initial_qpos=initial_qpos, reward_type=reward_type)
        os.remove(model_path)
        utils.EzPickle.__init__(self)
        self.size_object = self.sim.model.geom_size[self.sim.model.geom_name2id('object0')]
        self.size_obstacle = np.array([0.15, 0.15, 0.15])
        self.action_space = spaces.Box(-1., 1., shape=(8,), dtype='float32')
        self.initial_gripper_xpos_0 = self.sim.data.get_site_xpos('robot0:grip').copy()
        self.initial_gripper_xpos_1 = self.sim.data.get_site_xpos('robot1:grip').copy()
        self.initial_gripper_xpos = (self.initial_gripper_xpos_0 + self.initial_gripper_xpos_1) / 2
        self.maybe_gripper_xpos = [[1.1, 0.7], [1.5, 0.8]]
        self.maybe_gripper_fix_xpos = [[1.0, 0.75], [1.65, 0.75]]

    def _sample_goal(self):
        if not hasattr(self, 'size_object'):
            self.size_object = self.sim.model.geom_size[self.sim.model.geom_name2id('object0')]
        if not hasattr(self, 'current_nobject'):
            self.current_nobject = self.n_object
        if not hasattr(self, 'selected_objects'):
            self.selected_objects = np.arange(self.current_nobject)
        # if not hasattr(self, 'sample_easy'):
        #     self.sample_easy = False

        if self.task_mode == 1:
            if self.has_base:
                # base_nobject > 0
                # Randomize goal height here.
                goal_height = self.height_offset + np.random.randint(self._base_nobject, self.current_nobject) * 2 * \
                              self.size_object[2]
                goal = np.concatenate([self.maybe_goal_xy, [goal_height]])
                # g_idx = np.random.randint(self.current_nobject)
                g_idx = np.random.choice(self.selected_objects)
                _count = 0
                while abs(self.sim.data.get_joint_qpos('object%d:joint' % g_idx)[0] - self.maybe_goal_xy[0]) < \
                        self.size_object[0] \
                        and abs(self.sim.data.get_joint_qpos('object%d:joint' % g_idx)[1] - self.maybe_goal_xy[1]) < \
                        self.size_object[1]:
                    _count += 1
                    if _count > 100:
                        print(self.maybe_goal_xy, self.sim.data.get_joint_qpos('object0:joint'),
                              self.sim.data.get_joint_qpos('object1:joint'))
                        print(self.current_nobject, self.n_object)
                        raise RuntimeError
                    # g_idx = np.random.randint(self.current_nobject)
                    g_idx = np.random.choice(self.selected_objects)
            else:
                assert self._base_nobject == 0
                goal_height = self.height_offset + np.random.randint(0, self.current_nobject) * 2 * self.size_object[2]
                goal = np.concatenate([[self.np_random.uniform(1.1, 1.5),
                                        self.np_random.uniform(0.6, 0.9)], [goal_height]])
                # g_idx = np.random.randint(self.current_nobject)
                g_idx = np.random.choice(self.selected_objects)
            one_hot = np.zeros(self.n_object)
            one_hot[g_idx] = 1
        else:
            # Pick and place
            # g_idx = np.random.randint(self.current_nobject)
            g_idx = np.random.choice(self.selected_objects)
            if hasattr(self, 'has_base') and self.has_base:
                while abs(self.sim.data.get_joint_qpos('object%d:joint' % g_idx)[0] - self.maybe_goal_xy[0]) < 0.01 \
                        and abs(
                    self.sim.data.get_joint_qpos('object%d:joint' % g_idx)[1] - self.maybe_goal_xy[1]) < 0.01:
                    g_idx = np.random.choice(self.selected_objects)
            one_hot = np.zeros(self.n_object)
            one_hot[g_idx] = 1
            goal = np.array([self.np_random.uniform(1.1, 1.5),
                             self.np_random.uniform(0.6, 0.9),
                             self.height_offset])
            if self.np_random.uniform() < 0.5:
                goal[2] += self.np_random.uniform(0, 0.25)
        goal = np.concatenate([goal, one_hot])
        return goal.copy()

    def _reset_sim(self):
        self.sim.set_state(self.initial_state)
        if self._all:
            self.current_nrobot = self.n_robot
        else:
            self.current_nrobot, _ = self.task_robot_array[int(
                self.np_random.uniform() * len(self.task_robot_array))]
        self.selected_robots = np.random.choice(
            np.arange(self.n_robot), self.current_nrobot, replace=False)
        self.robot_onehot = np.zeros(self.n_robot)
        if self.random_gripper:
            for i in range(self.n_robot):
                if i in self.selected_robots:
                    self.robot_onehot[i] = 1
                    # mocap_pos = np.concatenate([self.np_random.uniform(self.maybe_gripper_xpos[i]), [0.5]])
                    mocap_pos = np.concatenate([self.np_random.uniform(low=[1.1, 0.7], high=[1.5, 0.8]), [0.5]])
                    self.sim.data.set_mocap_pos('robot' + str(i) + ':mocap', mocap_pos)
                else:
                    mocap_pos = np.array([self.maybe_gripper_fix_xpos[i] + [0.5]])
                    self.sim.data.set_mocap_pos('robot' + str(i) + ':mocap', mocap_pos)
            for _ in range(10):
                self.sim.step()
            self._step_callback()

        def is_valid(objects_xpos):
            for id1 in range(len(objects_xpos)):
                for id2 in range(id1 + 1, len(objects_xpos)):
                    if abs(objects_xpos[id1][0] - objects_xpos[id2][0]) < 2 * self.size_object[0] and \
                            abs(objects_xpos[id1][1] - objects_xpos[id2][1]) < 2 * self.size_object[1]:
                        return False
            return True

        if self.np_random.uniform() < self.random_ratio:
            self.task_mode = 0  # pick and place
        else:
            self.task_mode = 1
        # Randomize start position of object.
        if self.has_object:
            self.has_base = False
            task_rand = self.np_random.uniform()
            self.current_nobject, base_nobject = self.task_array[int(
                task_rand * len(self.task_array))]
            self._base_nobject = base_nobject
            self.selected_objects = np.random.choice(
                np.arange(self.n_object), self.current_nobject, replace=False)
            self.tower_height = self.height_offset + \
                                (base_nobject - 1) * self.size_object[2] * 2
            # if self.random_box and self.np_random.uniform() < self.random_ratio:
            if self.random_box:
                if base_nobject > 0:
                    self.has_base = True
                    objects_xpos = []
                    # self.maybe_goal_xy = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.obj_range,
                    #                                                                             self.obj_range, size=2)
                    self.maybe_goal_xy = np.array([self.np_random.uniform(1.1, 1.5), self.np_random.uniform(0.6, 0.9)])
                    for i in range(base_nobject):
                        objects_xpos.append(np.concatenate(
                            [self.maybe_goal_xy.copy(), [self.height_offset + i * 2 * self.size_object[2]]]))
                    for i in range(base_nobject, self.current_nobject):
                        objects_xpos.append(np.concatenate(
                            [np.array([self.np_random.uniform(1.1, 1.5), self.np_random.uniform(0.6, 0.9)]),
                             [self.height_offset]]))
                    while not is_valid(objects_xpos[base_nobject - 1:]):
                        for i in range(base_nobject, self.current_nobject):
                            objects_xpos[i][:2] = np.array(
                                [self.np_random.uniform(1.1, 1.5), self.np_random.uniform(0.6, 0.9)])
                    import random
                    random.shuffle(objects_xpos)
                else:
                    objects_xpos = []
                    for i in range(self.current_nobject):
                        objects_xpos.append(np.concatenate(
                            [np.array([self.np_random.uniform(1.1, 1.5), self.np_random.uniform(0.6, 0.9)]),
                             [self.height_offset]]))
                    while not is_valid(objects_xpos):
                        for i in range(self.current_nobject):
                            objects_xpos[i][:2] = np.array(
                                [self.np_random.uniform(1.1, 1.5), self.np_random.uniform(0.6, 0.9)])
            else:
                raise NotImplementedError

            for i in range(self.n_object):
                object_qpos = self.sim.data.get_joint_qpos(
                    'object%d:joint' % i)
                if i in self.selected_objects:
                    object_qpos[:3] = objects_xpos[np.where(
                        self.selected_objects == i)[0][0]]
                else:
                    object_qpos[:3] = np.array([-1 - i, -1, 0])
                self.sim.data.set_joint_qpos('object%d:joint' % i, object_qpos)
        self.sim.forward()
        return True


class DoublePick(DoubleFetchStackEnv):
    def __init__(self, reward_type='sparse', random_gripper=True, random_box=True, random_ratio=1.0, n_object=2,
                 n_obs=1, n_robot=2):
        initial_qpos = {
            'robot0:slide0': 0.0,
            'robot0:slide1': 0.0,
            'robot0:slide2': 0.0,

            'robot1:slide0': 0.0,
            'robot1:slide1': 0.0,
            'robot1:slide2': 0.0,

            'robot2:slide0': 0.0,
            'robot2:slide1': 0.0,
            'robot2:slide2': 0.0,

            'robot3:slide0': 0.0,
            'robot3:slide1': 0.0,
            'robot3:slide2': 0.0,
        }
        for i in range(n_object):
            initial_qpos['object%d:joint' % i] = [
                0.95 + 0.05 * i, 0.75 + 0.1 * (i - n_object // 2), 0.41, 1., 0., 0., 0.]
        self.random_gripper = random_gripper
        self.random_box = random_box
        self.random_ratio = random_ratio
        self.n_object = n_object
        self.n_robot = n_robot
        self.task_array = []
        for i in range(self.n_object):
            self.task_array.extend([(i + 1, j) for j in range(i + 1)])
        self.task_robot_array = []
        for i in range(self.n_robot):
            self.task_robot_array.extend([(i + 1, j) for j in range(i + 1)])
        with tempfile.NamedTemporaryFile(mode='wt',
                                         dir=os.path.join(os.path.dirname(
                                             __file__), '../assets', 'fetch'),
                                         delete=False, suffix=".xml") as fp:
            fp.write(generate_multi_xml(self.n_object))
            model_path = fp.name
        self.task_mode = 0  # 0: pick and place, 1: stack
        fetch_env.FetchEnv.__init__(
            self, model_path, has_object=True, block_gripper=False, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=True, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type=reward_type)
        os.remove(model_path)
        utils.EzPickle.__init__(self)
        self.size_object = self.sim.model.geom_size[self.sim.model.geom_name2id(
            'object0')]
        self.size_obstacle = np.array([0.15, 0.15, 0.15])
        self.action_space = spaces.Box(-1., 1., shape=(16,), dtype='float32')
        self.initial_gripper_xpos = self.sim.data.get_site_xpos('robot0:grip')
        for i in range(1, self.n_robot):
            self.initial_gripper_xpos += self.sim.data.get_site_xpos('robot' + str(i) + ':grip')
        self.initial_gripper_xpos /= self.n_robot
        self.n_obs = n_obs
        self.maybe_gripper_xpos = [[1.0, 0.5], [1.2, 1.0],
                                   [1.4, 0.5], [1.65, 1.0],
                                   [1.0, 0.5], [1.65, 0.65],
                                   [1.0, 0.85], [1.65, 1.0]]

        self.maybe_gripper_fix_xpos = [[1.0, 0.75], [1.65, 0.75],
                                       [1.4, 0.5], [1.4, 1.0]]

    def _sample_goal(self):
        if not hasattr(self, 'size_object'):
            self.size_object = self.sim.model.geom_size[self.sim.model.geom_name2id('object0')]
        if not hasattr(self, 'current_nobject'):
            self.current_nobject = self.n_object
        if not hasattr(self, 'selected_objects'):
            self.selected_objects = np.arange(self.current_nobject)
        # if not hasattr(self, 'sample_easy'):
        #     self.sample_easy = False

        if self.task_mode == 1:
            if self.has_base:
                # base_nobject > 0
                # Randomize goal height here.
                goal_height = self.height_offset + np.random.randint(self._base_nobject, self.current_nobject) * 2 * \
                              self.size_object[2]
                goal = np.concatenate([self.maybe_goal_xy, [goal_height]])
                # g_idx = np.random.randint(self.current_nobject)
                g_idx = np.random.choice(self.selected_objects)
                _count = 0
                while abs(self.sim.data.get_joint_qpos('object%d:joint' % g_idx)[0] - self.maybe_goal_xy[0]) < \
                        self.size_object[0] \
                        and abs(self.sim.data.get_joint_qpos('object%d:joint' % g_idx)[1] - self.maybe_goal_xy[1]) < \
                        self.size_object[1]:
                    _count += 1
                    if _count > 100:
                        print(self.maybe_goal_xy, self.sim.data.get_joint_qpos('object0:joint'),
                              self.sim.data.get_joint_qpos('object1:joint'))
                        print(self.current_nobject, self.n_object)
                        raise RuntimeError
                    # g_idx = np.random.randint(self.current_nobject)
                    g_idx = np.random.choice(self.selected_objects)
            else:
                assert self._base_nobject == 0
                goal_height = self.height_offset + np.random.randint(0, self.current_nobject) * 2 * self.size_object[2]
                goal = np.concatenate([[self.np_random.uniform(1.0, 1.6),
                                        self.np_random.uniform(0.5, 1.0)], [goal_height]])
                # g_idx = np.random.randint(self.current_nobject)
                g_idx = np.random.choice(self.selected_objects)
            one_hot = np.zeros(self.n_object)
            one_hot[g_idx] = 1
        else:
            # Pick and place
            # g_idx = np.random.randint(self.current_nobject)
            g_idx = np.random.choice(self.selected_objects)
            if hasattr(self, 'has_base') and self.has_base:
                while abs(self.sim.data.get_joint_qpos('object%d:joint' % g_idx)[0] - self.maybe_goal_xy[0]) < 0.01 \
                        and abs(
                    self.sim.data.get_joint_qpos('object%d:joint' % g_idx)[1] - self.maybe_goal_xy[1]) < 0.01:
                    g_idx = np.random.choice(self.selected_objects)
            one_hot = np.zeros(self.n_object)
            one_hot[g_idx] = 1
            goal = np.array([self.np_random.uniform(1.0, 1.6),
                             self.np_random.uniform(0.5, 1.0),
                             self.height_offset])
            if self.np_random.uniform() < 0.5:
                goal[2] += self.np_random.uniform(0, 0.25)
        goal = np.concatenate([goal, one_hot])
        return goal.copy()

    def _reset_sim(self):
        self.sim.set_state(self.initial_state)
        self.current_nrobot, _ = self.task_robot_array[int(
            self.np_random.uniform() * len(self.task_robot_array))]
        self.selected_robots = np.random.choice(
            np.arange(self.n_robot), self.current_nrobot, replace=False)
        self.robot_onehot = np.zeros(self.n_robot)
        if self.random_gripper:
            for i in range(self.n_robot):
                if i in self.selected_robots:
                    self.robot_onehot[i] = 1
                    mocap_pos = np.concatenate([self.np_random.uniform(self.maybe_gripper_xpos[i]), [0.5]])
                    self.sim.data.set_mocap_pos('robot' + str(i) + ':mocap', mocap_pos)
                else:
                    mocap_pos = np.array([self.maybe_gripper_fix_xpos[i] + [0.5]])
                    self.sim.data.set_mocap_pos('robot' + str(i) + ':mocap', mocap_pos)
            for _ in range(10):
                self.sim.step()
            self._step_callback()

        def is_valid(objects_xpos):
            for id1 in range(len(objects_xpos)):
                for id2 in range(id1 + 1, len(objects_xpos)):
                    if abs(objects_xpos[id1][0] - objects_xpos[id2][0]) < 2 * self.size_object[0] and \
                            abs(objects_xpos[id1][1] - objects_xpos[id2][1]) < 2 * self.size_object[1]:
                        return False
            return True

        if self.np_random.uniform() < self.random_ratio:
            self.task_mode = 0  # pick and place
        else:
            self.task_mode = 1
        # Randomize start position of object.
        if self.has_object:
            self.has_base = False
            task_rand = self.np_random.uniform()
            self.current_nobject, base_nobject = self.task_array[int(
                task_rand * len(self.task_array))]
            self._base_nobject = base_nobject
            self.selected_objects = np.random.choice(
                np.arange(self.n_object), self.current_nobject, replace=False)
            self.tower_height = self.height_offset + \
                                (base_nobject - 1) * self.size_object[2] * 2
            # if self.random_box and self.np_random.uniform() < self.random_ratio:
            if self.random_box:
                if base_nobject > 0:
                    self.has_base = True
                    objects_xpos = []
                    # self.maybe_goal_xy = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.obj_range,
                    #                                                                             self.obj_range, size=2)
                    self.maybe_goal_xy = np.array([self.np_random.uniform(1.0, 1.6), self.np_random.uniform(0.5, 1.0)])
                    for i in range(base_nobject):
                        objects_xpos.append(np.concatenate(
                            [self.maybe_goal_xy.copy(), [self.height_offset + i * 2 * self.size_object[2]]]))
                    for i in range(base_nobject, self.current_nobject):
                        objects_xpos.append(np.concatenate(
                            [np.array([self.np_random.uniform(1.0, 1.6), self.np_random.uniform(0.5, 1.0)]),
                             [self.height_offset]]))
                    while not is_valid(objects_xpos[base_nobject - 1:]):
                        for i in range(base_nobject, self.current_nobject):
                            objects_xpos[i][:2] = np.array(
                                [self.np_random.uniform(1.0, 1.6), self.np_random.uniform(0.5, 1.0)])
                    import random
                    random.shuffle(objects_xpos)
                else:
                    objects_xpos = []
                    for i in range(self.current_nobject):
                        objects_xpos.append(np.concatenate(
                            [np.array([self.np_random.uniform(1.0, 1.6), self.np_random.uniform(0.5, 1.0)]),
                             [self.height_offset]]))
                    while not is_valid(objects_xpos):
                        for i in range(self.current_nobject):
                            objects_xpos[i][:2] = np.array(
                                [self.np_random.uniform(1.0, 1.6), self.np_random.uniform(0.5, 1.0)])
            else:
                raise NotImplementedError

            for i in range(self.n_object):
                object_qpos = self.sim.data.get_joint_qpos(
                    'object%d:joint' % i)
                if i in self.selected_objects:
                    object_qpos[:3] = objects_xpos[np.where(
                        self.selected_objects == i)[0][0]]
                else:
                    object_qpos[:3] = np.array([-1 - i, -1, 0])
                self.sim.data.set_joint_qpos('object%d:joint' % i, object_qpos)
        self.sim.forward()
        return True

    def _set_action(self, action):
        assert action.shape == (16,)
        action = action.copy()  # ensure that we don't change the action outside of this scope
        pos_ctrl_0, gripper_ctrl_0 = action[:3], action[3]
        pos_ctrl_1, gripper_ctrl_1 = action[4:7], action[7]
        pos_ctrl_2, gripper_ctrl_2 = action[8:11], action[11]
        pos_ctrl_3, gripper_ctrl_3 = action[12:15], action[15]
        pos_ctrl_0 *= 0.05
        pos_ctrl_1 *= 0.05
        pos_ctrl_2 *= 0.05
        pos_ctrl_3 *= 0.05
        rot_ctrl_0 = np.array([1., 0., 1., 0.])
        rot_ctrl_1 = np.array([1., 0., 1., 0.])
        rot_ctrl_2 = np.array([1., 0., 1., 0.])
        rot_ctrl_3 = np.array([1., 0., 1., 0.])
        gripper_ctrl_0 = np.array([gripper_ctrl_0, gripper_ctrl_0])
        gripper_ctrl_1 = np.array([gripper_ctrl_1, gripper_ctrl_1])
        gripper_ctrl_2 = np.array([gripper_ctrl_2, gripper_ctrl_2])
        gripper_ctrl_3 = np.array([gripper_ctrl_3, gripper_ctrl_3])
        assert gripper_ctrl_0.shape == (2,)
        assert gripper_ctrl_1.shape == (2,)
        assert gripper_ctrl_2.shape == (2,)
        assert gripper_ctrl_3.shape == (2,)
        if self.block_gripper:
            gripper_ctrl_0 = np.zeros_like(gripper_ctrl_0)
            gripper_ctrl_1 = np.zeros_like(gripper_ctrl_1)
            gripper_ctrl_2 = np.zeros_like(gripper_ctrl_2)
            gripper_ctrl_3 = np.zeros_like(gripper_ctrl_3)

        action = np.concatenate([pos_ctrl_0, rot_ctrl_0, pos_ctrl_1,
                                 rot_ctrl_1, pos_ctrl_2, rot_ctrl_2, pos_ctrl_3,
                                 rot_ctrl_3, gripper_ctrl_0, gripper_ctrl_1,
                                 gripper_ctrl_2, gripper_ctrl_3])

        # Apply action to simulation.
        ctrl_set_action(self.sim, action)
        mocap_set_action(self.sim, action)

    def _env_setup(self, initial_qpos):
        for name, value in initial_qpos.items():
            self.sim.data.set_joint_qpos(name, value)
        robotics_utils.reset_mocap_welds(self.sim)
        self.sim.forward()

        # Move end effector into position.
        gripper_target_0 = np.array([-0.498, 0.005, -0.431 + self.gripper_extra_height]) + self.sim.data.get_site_xpos(
            'robot0:grip')
        gripper_target_1 = np.array([-0.498, 0.005, -0.431 + self.gripper_extra_height]) + self.sim.data.get_site_xpos(
            'robot1:grip')
        gripper_target_2 = np.array([-0.498, 0.005, -0.431 + self.gripper_extra_height]) + self.sim.data.get_site_xpos(
            'robot0:grip')
        gripper_target_3 = np.array([-0.498, 0.005, -0.431 + self.gripper_extra_height]) + self.sim.data.get_site_xpos(
            'robot1:grip')
        gripper_rotation_0 = np.array([0., 0., 0., 0.])
        gripper_rotation_1 = np.array([-0.7, 0., 0., 1.])
        gripper_rotation_2 = np.array([1., 0., 0., 1.])
        gripper_rotation_3 = np.array([-1, 0., 0., 1.])
        self.sim.data.set_mocap_pos('robot0:mocap', gripper_target_0)
        self.sim.data.set_mocap_quat('robot0:mocap', gripper_rotation_0)
        self.sim.data.set_mocap_pos('robot1:mocap', gripper_target_1)
        self.sim.data.set_mocap_quat('robot1:mocap', gripper_rotation_1)
        self.sim.data.set_mocap_pos('robot2:mocap', gripper_target_2)
        self.sim.data.set_mocap_quat('robot2:mocap', gripper_rotation_2)
        self.sim.data.set_mocap_pos('robot3:mocap', gripper_target_3)
        self.sim.data.set_mocap_quat('robot3:mocap', gripper_rotation_3)
        for _ in range(10):
            self.sim.step()

        self.initial_gripper_xpos = self.sim.data.get_site_xpos('robot0:grip')
        # Extract information for sampling goals.
        for i in range(1, self.n_robot):
            self.initial_gripper_xpos += self.sim.data.get_site_xpos('robot' + str(i) + ':grip')
        self.initial_gripper_xpos /= self.n_robot
        if self.has_object:
            self.height_offset = 0.42

    def _get_obs(self):
        grip_pos_0 = self.sim.data.get_site_xpos('robot0:grip')
        grip_pos_1 = self.sim.data.get_site_xpos('robot1:grip')
        grip_pos_2 = self.sim.data.get_site_xpos('robot2:grip')
        grip_pos_3 = self.sim.data.get_site_xpos('robot3:grip')
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp_0 = self.sim.data.get_site_xvelp('robot0:grip') * dt
        grip_velp_1 = self.sim.data.get_site_xvelp('robot1:grip') * dt
        grip_velp_2 = self.sim.data.get_site_xvelp('robot2:grip') * dt
        grip_velp_3 = self.sim.data.get_site_xvelp('robot3:grip') * dt
        robot_qpos, robot_qvel = robotics_utils.robot_get_obs(self.sim)
        if self.has_object:
            object_pos = [self.sim.data.get_site_xpos('object' + str(i)) if i in self.selected_objects else np.zeros(3)
                          for i in range(self.n_object)]
            # object_pos = [self.sim.data.get_site_xpos('object' + str(i)) for i in range(self.current_nobject)] \
            #              + [np.zeros(3) for _ in range(self.current_nobject, self.n_object)]
            # rotations
            object_rot = [rotations.mat2euler(self.sim.data.get_site_xmat('object' + str(i)))
                          if i in self.selected_objects else np.zeros(3) for i in range(self.n_object)]
            # object_rot = [rotations.mat2euler(self.sim.data.get_site_xmat('object' + str(i))) for i in range(self.current_nobject)] \
            #              + [np.zeros(3) for _ in range(self.current_nobject, self.n_object)]
            # velocities
            object_velp = [self.sim.data.get_site_xvelp('object' + str(i)) * dt
                           if i in self.selected_objects else np.zeros(3) for i in range(self.n_object)]
            # object_velp = [self.sim.data.get_site_xvelp('object' + str(i)) * dt for i in range(self.current_nobject)] \
            #               + [np.zeros(3) for _ in range(self.current_nobject, self.n_object)]
            object_velr = [self.sim.data.get_site_xvelr('object' + str(i)) * dt
                           if i in self.selected_objects else np.zeros(3) for i in range(self.n_object)]
            # object_velr = [self.sim.data.get_site_xvelr('object' + str(i)) * dt for i in range(self.current_nobject)] \
            #               + [np.zeros(3) for _ in range(self.current_nobject, self.n_object)]
            # gripper state
            # object_rel_pos = [pos - grip_pos for pos in object_pos]
            object_rel_pos_0 = [object_pos[i] - grip_pos_0 if i in self.selected_objects else np.zeros(3) for i in
                                range(self.n_object)]
            object_rel_pos_1 = [object_pos[i] - grip_pos_1 if i in self.selected_objects else np.zeros(3) for i in
                                range(self.n_object)]
            object_rel_pos_2 = [object_pos[i] - grip_pos_2 if i in self.selected_objects else np.zeros(3) for i in
                                range(self.n_object)]
            object_rel_pos_3 = [object_pos[i] - grip_pos_3 if i in self.selected_objects else np.zeros(3) for i in
                                range(self.n_object)]
            # object_rel_pos = [object_pos[i] - grip_pos for i in range(self.current_nobject)] \
            #                  + [np.zeros(3) for _ in range(self.current_nobject, self.n_object)]
            # object_velp = [velp - grip_velp for velp in object_velp]
            object_velp_0 = [object_velp[i] - grip_velp_0 if i in self.selected_objects else np.zeros(3) for i in
                             range(self.n_object)]
            object_velp_1 = [object_velp[i] - grip_velp_1 if i in self.selected_objects else np.zeros(3) for i in
                             range(self.n_object)]
            object_velp_2 = [object_velp[i] - grip_velp_2 if i in self.selected_objects else np.zeros(3) for i in
                             range(self.n_object)]
            object_velp_3 = [object_velp[i] - grip_velp_3 if i in self.selected_objects else np.zeros(3) for i in
                             range(self.n_object)]
            # object_velp = [object_velp[i] - grip_velp for i in range(self.current_nobject)] \
            #               + [np.zeros(3) for _ in range(self.current_nobject, self.n_object)]

            object_pos = np.concatenate(object_pos)
            object_rot = np.concatenate(object_rot)
            object_velp_0 = np.concatenate(object_velp_0)
            object_velp_1 = np.concatenate(object_velp_1)
            object_velp_2 = np.concatenate(object_velp_2)
            object_velp_3 = np.concatenate(object_velp_3)
            object_velr = np.concatenate(object_velr)
            object_rel_pos_0 = np.concatenate(object_rel_pos_0)
            object_rel_pos_1 = np.concatenate(object_rel_pos_1)
            object_rel_pos_2 = np.concatenate(object_rel_pos_2)
            object_rel_pos_3 = np.concatenate(object_rel_pos_3)

        else:
            object_pos = object_rot = object_velp_0 = object_velp_1 = object_velp_2 = object_velp_3 = object_velr = object_rel_pos_0 \
                = object_rel_pos_1 = object_rel_pos_2 \
                = object_rel_pos_3 = np.zeros(0)
        gripper_state_0 = robot_qpos[13:15]
        gripper_state_1 = robot_qpos[29:30]
        gripper_vel_0 = robot_qvel[13:15] * dt
        gripper_vel_1 = robot_qvel[29:30] * dt
        gripper_state_2 = robot_qpos[44:45]
        gripper_state_3 = robot_qpos[-2:]
        gripper_vel_2 = robot_qvel[44:45] * dt
        gripper_vel_3 = robot_qvel[-2:] * dt

        if not self.has_object:
            achieved_goal = np.concatenate([grip_pos_0.copy(), grip_pos_1.copy(),
                                            grip_pos_2.copy(), grip_pos_3.copy()])
        else:
            one_hot = self.goal[3:]
            idx = np.argmax(one_hot)
            achieved_goal = np.concatenate(
                [object_pos[3 * idx: 3 * (idx + 1)], one_hot])
        task_one_hot = np.zeros(2)
        task_one_hot[self.task_mode] = 1
        obs = np.concatenate([
            object_pos.ravel(), object_rot.ravel(),
            object_velr.ravel(),
            grip_pos_0, object_rel_pos_0.ravel(
            ), gripper_state_0,
            object_velp_0.ravel(), grip_velp_0, gripper_vel_0, grip_pos_1, object_rel_pos_1.ravel(
            ), gripper_state_1,
            object_velp_1.ravel(), grip_velp_1, gripper_vel_1, grip_pos_2, object_rel_pos_2.ravel(
            ), gripper_state_2,
            object_velp_2.ravel(), grip_velp_2, gripper_vel_2, grip_pos_3, object_rel_pos_3.ravel(
            ), gripper_state_3,
            object_velp_3.ravel(), grip_velp_3, gripper_vel_3, task_one_hot
        ])
        return {
            'observation': obs.copy(),
            # 'achieved_goal': achieved_goal.copy(),
            'achieved_goal': achieved_goal.copy(),
            # 'desired_goal': self.goal.copy(),
            'desired_goal': self.goal.copy(),
        }


if __name__ == '__main__':
    env = DoubleFetchStackEnv_v2(n_object=1,
                                 random_ratio=0.0)  # env = DoubleFetchStackEnv_v2(n_object=6, random_ratio=0.0)
    obs = env.reset()
    # while env.current_nobject != 6:
    #     obs = env.reset()
    for _ in range(10000):
        env.step(2 * np.random.rand(8) - 1.)
        # env.step(np.zeros(8))
        env.render()

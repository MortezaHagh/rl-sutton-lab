from dataclasses import dataclass
import numpy as np
from collections import defaultdict
from .utils import argmax_random_tie


class BaseEnv:
    def __init__(self):
        self.goal_state = None
        self.start_state = None
        self.n_actions = None


@dataclass
class NstepTDHyperparameters:
    '''hyperparameters for N-step TD algorithms '''
    n: int = 5
    alpha: float = 0.1
    gamma: float = 1.0
    epsilon: float = 0.1
    n_episodes: int = 100
    excepted_sarsa: bool = False
    behavior_epsilon: float = 0.2


class NstepTDBase:
    '''base class for N-step TD algorithms, contains common methods and attributes '''

    def __init__(self, env: BaseEnv, hyperparameters: NstepTDHyperparameters):
        self.env = env
        self.p = hyperparameters
        self.n_steps_per_episode = []
        # initialize Q-values to zero
        # self.Q = defaultdict(lambda: np.zeros(self.env.n_actions))
        # random initialization of Q-values to encourage exploration, normal dist
        self.Q = defaultdict(lambda: np.random.rand(self.env.n_actions))
        self.Q[self.env.goal_state] = np.zeros(self.env.n_actions)  # Q-values for goal state are zero

    def get_expected_return(self, states, t):
        expected_Q = np.dot(self.Q[states[t]], self.get_greedy_action_probabilities(states[t]))
        return expected_Q

    def step(self, state, action):
        '''take action and return next state, reward, and done flag'''
        x, y = state
        dx, dy = self.env.actions[action]
        # length
        l = np.sqrt(dx ** 2 + dy ** 2)
        wind_strength = self.env.wind[x]
        x += dx
        y += dy
        y += wind_strength
        # keep within grid boundaries
        x = max(0, min(self.env.x_max - 1, x))
        y = max(0, min(self.env.y_max - 1, y))
        next_state = (x, y)
        done = next_state == self.env.goal_state
        reward = 0 if done else -l
        return next_state, reward, done

    def greedy_policy(self, state):
        '''epsilon-greedy action selection '''
        p_other = self.p.epsilon / self.env.n_actions
        p_best = 1 - self.p.epsilon + p_other
        if np.random.rand() < self.p.epsilon:
            return np.random.choice(self.env.n_actions), p_other
        else:
            return argmax_random_tie(self.Q[state]), p_best

    def behavior_policy(self, state):
        p_other = self.p.behavior_epsilon / self.env.n_actions
        p_best = 1 - self.p.behavior_epsilon + p_other
        if np.random.rand() < self.p.behavior_epsilon:
            action = np.random.choice(self.env.n_actions)
        else:
            action = argmax_random_tie(self.Q[state])
        p_b = p_best if action == argmax_random_tie(self.Q[state]) else p_other
        return action, p_b

    def get_greedy_action_probabilities(self, state):
        '''get action probabilities for epsilon-greedy policy'''
        action_probabilities = np.ones(self.env.n_actions) * self.p.epsilon / self.env.n_actions
        best_action = argmax_random_tie(self.Q[state])
        action_probabilities[best_action] += (1.0 - self.p.epsilon)
        return action_probabilities

    def get_action_probability(self, state, action):
        '''get probability of taking action under epsilon-greedy policy'''
        if action == argmax_random_tie(self.Q[state]):
            return 1 - self.p.epsilon + (self.p.epsilon / self.env.n_actions)
        else:
            return self.p.epsilon / self.env.n_actions

    def rho_action(self, state, action, p_b):
        '''calculate importance sampling ratio for given state and action'''
        target_probabilities = self.get_greedy_action_probabilities(state)
        p_t = target_probabilities[action]
        if p_t == 0:
            return 0.0  # if target policy never takes this action, ratio is zero
        rho = p_t / (p_b + 1e-8)  # add small constant to avoid division by zero
        rho = min(max(rho, 0), 10)  # cap rho to prevent extreme values
        return rho

    # This method is a placeholder and should be implemented in the subclasses
    def run(self):
        raise NotImplementedError("Subclasses should implement this method")


class NstepTDOnPolicy(NstepTDBase):
    def __init__(self, env: BaseEnv, hyperparameters: NstepTDHyperparameters):
        super().__init__(env, hyperparameters)

    def run(self):
        for episode in range(self.p.n_episodes):
            state = self.env.start_state
            action, _ = self.greedy_policy(state)
            states = [state]
            actions = [action]
            rewards = [0]  # reward for time step 0 is 0
            t = 0
            T = float('inf')
            while True:
                if t < T:
                    next_state, reward, done = self.step(states[-1], actions[-1])
                    states.append(next_state)
                    rewards.append(reward)
                    if done:
                        T = t + 1
                    else:
                        next_action, _ = self.greedy_policy(next_state)
                        actions.append(next_action)
                tau = t - self.p.n + 1
                if tau >= 0:
                    G = sum(self.p.gamma ** (i - tau - 1) * rewards[i] for i in range(tau + 1, min(tau + self.p.n, T) + 1))
                    if (tau + self.p.n) < T:
                        if self.p.excepted_sarsa:
                            G += self.p.gamma ** self.p.n * self.get_expected_return(states, tau + self.p.n)
                        else:
                            G += self.p.gamma ** self.p.n * self.Q[states[tau + self.p.n]][actions[tau + self.p.n]]
                    state_tau = states[tau]
                    action_tau = actions[tau]
                    # update Q-value
                    self.Q[state_tau][action_tau] += self.p.alpha * (G - self.Q[state_tau][action_tau])
                if tau == T - 1:
                    break
                t += 1
            self.n_steps_per_episode.append(t)


class NstepTDOffPolicy(NstepTDBase):
    def __init__(self, env: BaseEnv, hyperparameters: NstepTDHyperparameters):
        super().__init__(env, hyperparameters)

    def run(self):
        for episode in range(self.p.n_episodes):
            state = self.env.start_state
            action, p_b = self.behavior_policy(state)
            states = [state]
            actions = [action]
            rewards = [0]  # reward for time step 0 is 0
            rhos = [self.rho_action(state, action, p_b)]
            t = 0
            T = float('inf')
            while True:
                if t < T:
                    next_state, reward, done = self.step(states[-1], actions[-1])
                    states.append(next_state)
                    rewards.append(reward)
                    if done:
                        T = t + 1
                    else:
                        action, p_b = self.behavior_policy(next_state)
                        actions.append(action)
                        rhos.append(self.rho_action(next_state, action, p_b))
                tau = t - self.p.n + 1
                if tau >= 0:
                    G = sum(self.p.gamma ** (i - tau - 1) * rewards[i] for i in range(tau + 1, min(tau + self.p.n, T) + 1))
                    rho_last_index = min(tau + self.p.n - (1 if self.p.excepted_sarsa else 0), T - 1)
                    if rho_last_index >= tau + 1:
                        RHO = np.prod(rhos[tau + 1:rho_last_index + 1])
                    else:
                        RHO = 1.0
                    # print(f"Episode {episode}, time {t}, tau {tau}, G {G:.2f}, RHO {RHO:.2f}")
                    if (tau + self.p.n) < T:
                        if self.p.excepted_sarsa:
                            G += self.p.gamma ** self.p.n * self.get_expected_return(states, tau + self.p.n)
                        else:
                            G += self.p.gamma ** self.p.n * self.Q[states[tau + self.p.n]][actions[tau + self.p.n]]
                    state_tau = states[tau]
                    action_tau = actions[tau]
                    # update Q-value
                    self.Q[state_tau][action_tau] += self.p.alpha * RHO * (G - self.Q[state_tau][action_tau])
                if tau == T - 1:
                    break
                t += 1
            self.n_steps_per_episode.append(t)


class NstepTDOffPolicyRecursive(NstepTDOffPolicy):
    def __init__(self, env: BaseEnv, hyperparameters: NstepTDHyperparameters):
        super().__init__(env, hyperparameters)

    def run(self):
        def rect_return(tau, h):
            if tau + 1 >= T:
                return rewards[T]
            if tau == h:
                return self.Q[states[tau]][actions[tau]]
            new_tau = tau + 1
            rho = rhos[new_tau]
            return rewards[new_tau] + self.p.gamma * rho * (rect_return(new_tau, h) - self.Q[states[new_tau]][actions[new_tau]]) + self.p.gamma * self.get_expected_return(states, new_tau)

        for episode in range(self.p.n_episodes):
            state = self.env.start_state
            action, p_b = self.behavior_policy(state)
            states = [state]
            actions = [action]
            rewards = [0]  # reward for time step 0 is 0
            rhos = [self.rho_action(state, action, p_b)]
            t = 0
            T = float('inf')
            while True:
                if t < T:
                    next_state, reward, done = self.step(states[-1], action)
                    states.append(next_state)
                    rewards.append(reward)
                    if done:
                        T = t + 1
                    else:
                        action, p_b = self.behavior_policy(next_state)
                        actions.append(action)
                        rhos.append(self.rho_action(next_state, action, p_b))
                tau = t - self.p.n + 1
                if tau >= 0 and tau < T - 1:
                    rho = rhos[tau + 1]
                    G = rect_return(tau, min(tau + self.p.n, T - 1))
                    state_tau = states[tau]
                    action_tau = actions[tau]
                    # update Q-value
                    self.Q[state_tau][action_tau] += self.p.alpha * rho * (G - self.Q[state_tau][action_tau])
                if tau == T - 1:
                    break
                t += 1
            self.n_steps_per_episode.append(t)


class NstepTDTreeBackup(NstepTDOffPolicy):
    def __init__(self, env: BaseEnv, hyperparameters: NstepTDHyperparameters):
        super().__init__(env, hyperparameters)

    def run(self):
        for episode in range(self.p.n_episodes):
            state = self.env.start_state
            action, _ = self.greedy_policy(state)
            states = [state]
            actions = [action]
            rewards = [0]  # reward for time step 0 is 0
            t = 0
            T = float('inf')
            while True:
                if t < T:
                    next_state, reward, done = self.step(states[-1], actions[-1])
                    states.append(next_state)
                    rewards.append(reward)
                    if done:
                        T = t + 1
                    else:
                        next_action, _ = self.greedy_policy(next_state)
                        actions.append(next_action)
                tau = t - self.p.n + 1
                if tau >= 0:
                    if (t + 1) >= T:
                        G = rewards[T]
                    else:
                        G = rewards[t + 1] + self.p.gamma * np.dot(self.Q[states[t + 1]], self.get_greedy_action_probabilities(states[t + 1]))
                    for k in range(min(t, T - 1), tau, -1):
                        next_state = states[k]
                        next_action = actions[k]
                        action_probabilities = self.get_greedy_action_probabilities(next_state)
                        expected_q = np.dot(self.Q[next_state], action_probabilities)
                        G = rewards[k] + self.p.gamma * (
                            expected_q
                            - action_probabilities[next_action] * self.Q[next_state][next_action]
                            + action_probabilities[next_action] * G
                        )
                    self.Q[states[tau]][actions[tau]] += self.p.alpha * (G - self.Q[states[tau]][actions[tau]])
                if tau == T - 1:
                    break
                t += 1
            self.n_steps_per_episode.append(t)


class NstepTDQSigma(NstepTDOffPolicy):
    def __init__(self, env: BaseEnv, hyperparameters: NstepTDHyperparameters):
        super().__init__(env, hyperparameters)

    def run(self):
        p_sigma = 0.5  # probability of using expected update vs sample update

        for episode in range(self.p.n_episodes):
            state = self.env.start_state
            action, p_b = self.behavior_policy(state)
            states = [state]
            actions = [action]
            rewards = [0]  # reward for time step 0 is 0
            rhos = [self.rho_action(state, action, p_b)]
            p_bs = [p_b]  # policy probabilities for each action
            T = float('inf')
            t = 0
            while True:
                if t < T:
                    next_state, reward, done = self.step(states[-1], actions[-1])
                    states.append(next_state)
                    rewards.append(reward)
                    if done:
                        T = t + 1
                    else:
                        action, p_b = self.behavior_policy(next_state)
                        actions.append(action)
                        rhos.append(self.rho_action(next_state, action, p_b))
                        p_bs.append(p_b)
                tau = t - self.p.n + 1
                if tau >= 0:
                    G = 0
                    for k in range(min(t + 1, T), tau, -1):
                        if k == T:
                            G = rewards[T]
                        else:
                            sigma = np.random.rand() < p_sigma
                            expected_q = self.get_expected_return(states, k)
                            G = rewards[k] \
                                + self.p.gamma * (sigma * rhos[k] + (1 - sigma) * p_bs[k]) * (G - self.Q[states[k]][actions[k]]) \
                                + self.p.gamma * expected_q
                    self.Q[states[tau]][actions[tau]] += self.p.alpha * (G - self.Q[states[tau]][actions[tau]])
                if tau == T - 1:
                    break
                t += 1
            self.n_steps_per_episode.append(t)

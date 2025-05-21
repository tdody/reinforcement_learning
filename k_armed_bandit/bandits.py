from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class OneArmedBandit:
    def __init__(self, mu: float, sigma: float, optimist_initial_value: float = 0):
        """
        Initialize the one-arm bandit with a given probability of success.

        Args:
            mu (float): Mean of the normal distribution.
            sigma (float): Standard deviation of the normal distribution.
            optimist_initial_value (float): Initial value of the bandit.
        """
        self.mu = mu
        self.sigma = sigma
        self.n = 0

        self.q = optimist_initial_value

    def pull(self, alpha: Optional[float] = None) -> float:
        """
        Simulate pulling the arm of the bandit.

        Args:
            alpha (float): Learning rate for updating the estimated value of the bandit.
            If None, the learning rate is set to 1/n where n is the number of times

        Returns:
            float: The reward received from pulling the arm.
        """
        # Simulate pulling the arm of the bandit
        reward = np.random.normal(self.mu, self.sigma)

        # Update the number of times the bandit has been pulled and the total reward
        self.n += 1

        # If alpha is None, set it to 1/n
        if alpha is None:
            alpha = 1 / self.n

        # Update the estimated value of the bandit using incremental mean formula
        self.q = self.q + (reward - self.q) * alpha

        return reward

    def __repr__(self) -> str:
        """
        String representation of the OneArmedBandit.
        Returns:
            str: String representation of the OneArmedBandit.
        """
        return f"OneArmedBandit(mu={self.mu:.2f}, sigma={self.sigma:.2f}, n={self.n}, q={self.q:.2f})"

    @classmethod
    def random_bandit(
        cls, optimist_initial_value: float = 0, random_seed: Optional[int] = None
    ) -> "OneArmedBandit":
        """
        Create a random one-arm bandit by sampling mu from N(0, 1) setting sigma = 1.0.

        Returns:
            OneArmedBandit: A random one-arm bandit.
        """
        if random_seed is not None:
            np.random.seed(random_seed)
        else:
            np.random.seed()
        mu = np.random.normal(0, 1)
        sigma = 1.0
        return cls(mu, sigma, optimist_initial_value)


class KArmedBanditExperiment:
    def __init__(self, k: int, optimist_initial_value: float = 0):
        """
        Initialize the K-armed bandit experiment with a given number of arms.
        Args:
            k (int): Number of arms.
            optimist_initial_value (float): Initial value of the bandit.
        """

        self.k = k
        self.bandits = [
            OneArmedBandit.random_bandit(optimist_initial_value) for _ in range(k)
        ]
        self.optimist_initial_value = optimist_initial_value

    def reset_bandits(self) -> None:
        """
        Reset the bandits to their initial state.
        Returns:
            None
        """
        for bandit in self.bandits:
            bandit.n = 0
            bandit.q = self.optimist_initial_value

    def __exploit(self) -> OneArmedBandit:
        """
        Exploit the best bandit from a list of bandits.

        Returns:
            OneArmedBandit: The best bandit.
        """
        return max(self.bandits, key=lambda b: b.q)

    def __explore(self) -> OneArmedBandit:
        """
        Explore a random bandit from a list of bandits.

        Returns:
            OneArmedBandit: A random bandit.
        """
        return np.random.choice(self.bandits)

    def epsilon_greedy(self, epsilon: float) -> tuple[float, bool]:
        """
        Choose a bandit using epsilon-greedy strategy.

        Args:
            epsilon (float): Probability of exploring.

        Returns:
            OneArmedBandit: The chosen bandit.
            boolean: True if the selected bandit is optimal, False otherwise.
        """
        selected_bandit: OneArmedBandit
        if np.random.rand() < epsilon:
            selected_bandit = self.__explore()
        else:
            selected_bandit = self.__exploit()

        return selected_bandit.pull(), is_optimal(self.bandits, selected_bandit)


def is_optimal(bandits: list[OneArmedBandit], selected_bandit: OneArmedBandit) -> bool:
    """
    Check if the selected bandit is optimal.

    Args:
        bandits (list[OneArmedBandit]): List of OneArmedBandit objects.
        selected_bandit (OneArmedBandit): The selected bandit.

    Returns:
        bool: True if the selected bandit is optimal, False otherwise.
    """
    return selected_bandit.mu == max(b.mu for b in bandits)


class ExperimentResult:
    epsilon: float
    """Epsilon value used in the experiment."""
    average_rewards: list[float]
    """Average rewards received across all simulations at a given action.
    average_rewards[i] = average reward at action i across all (n) experiments.
    """
    optimal_action_pct: list[float]
    """Percentage of optimal actions taken.
    optimal_action_pct[i] = percentage of optimal actions taken at action i across all (n) experiments.
    """
    n_experiments: int = 0
    """Number of times the experiment has been run."""

    def __init__(self, epsilon: float, n_actions: int):
        self.epsilon = epsilon
        self.average_rewards = []
        self.optimal_action_pct = []
        self.n_experiments = 0
        self.n_actions = n_actions

    def update(self, action_rewards: list[float], optimal_actions: list[bool]) -> None:
        """
        Update the average rewards and optimal action percentage.

        Args:
            action_rewards (list[float]): List of length (n) containing the rewards received across a single experiment.
            optimal_actions (list[bool]): List of length (n) containing the booleans indicating whether an action was optimal or not.

        Returns:
            None
        """
        if self.n_actions != len(action_rewards):
            raise ValueError(
                f"action_rewards must be of length {self.n_actions}, but got {len(action_rewards)}"
            )
        if self.n_actions != len(optimal_actions):
            raise ValueError(
                f"optimal_actions must be of length {self.n_actions}, but got {len(optimal_actions)}"
            )
        if self.n_experiments == 0:
            # Initialize the average rewards and optimal action percentage
            self.average_rewards = [0] * self.n_actions
            self.optimal_action_pct = [0] * self.n_actions

        self.average_rewards = [
            (self.average_rewards[i] * self.n_experiments) + action_rewards[i]
            for i in range(self.n_actions)
        ]
        self.optimal_action_pct = [
            (self.optimal_action_pct[i] * self.n_experiments) + optimal_actions[i]
            for i in range(self.n_actions)
        ]
        # Update the number of experiments
        self.n_experiments += 1

        # Normalize the average rewards and optimal action percentage
        self.average_rewards = [x / self.n_experiments for x in self.average_rewards]
        self.optimal_action_pct = [
            x / self.n_experiments for x in self.optimal_action_pct
        ]


def run_experiments(
    k_armed_bandit: KArmedBanditExperiment,
    epsilons: list[float],
    n_actions: int,
    n_experiments: int,
) -> dict[float, ExperimentResult]:
    """
    Run an experiment with a list of bandits using epsilon-greedy strategy.

    Args:
        k_armed_bandit (KArmedBanditExperiment): The K-armed bandit experiment.
        epsilons (list[float]): List of epsilon values for exploration.
        n_actions (int): Number of times to pull the bandit.
        n_experiments (int): Number of time we repeat teh experiment.

    Returns:
        list[ExperimentResult]: A list of ExperimentResult objects containing the
        epsilon value, average rewards, and optimal action percentage.
    """
    results = {epsilon: ExperimentResult(epsilon, [], [], 0) for epsilon in epsilons}

    for epsilon in epsilons:
        for _ in range(n_experiments):
            step_rewards: list[float] = []
            optimal_actions: list[bool] = []
            # reset the bandits for each experiment
            k_armed_bandit.reset_bandits()

            for _ in range(n_actions):
                step_reward, is_optimal = k_armed_bandit.epsilon_greedy(epsilon)
                step_rewards.append(step_reward)
                optimal_actions.append(is_optimal)

            # update the average rewards and optimal action percentage
            results[epsilon].update(step_rewards, optimal_actions)

    return results


def plot_average_rewards(
    rewards: list[dict[float, list[float]]], epsilons: list[float]
) -> None:
    """
    Plot the average rewards for each epsilon value.

    Args:
        rewards (list[dict[float, list[float]]]): A list of dictionaries where each
        dictionary contains the epsilon value as the key and a list of rewards as the value.
        epsilons (list[float]): List of epsilon values for exploration.

    Returns:
        None
    """
    sns.set(style="white")
    for eps in epsilons:
        avg_rewards = np.mean([r[eps] for r in rewards], axis=0)
        plt.plot(avg_rewards, label=f"epsilon={eps}")

    n_experiments = len(rewards)

    plt.xlabel("Steps")
    plt.ylabel(f"Average Reward for {n_experiments} experiments")
    plt.title("Epsilon-Greedy Strategy")
    plt.legend()
    plt.show()


def main():
    # Create a list of bandits
    bandits = [OneArmedBandit.random() for _ in range(10)]
    # Run the experiment with epsilon-greedy strategy
    run_experiment(bandits, epsilons=[0.0, 0.01, 0.1], n=1000)


if __name__ == "__main__":
    main()

from dataclasses import dataclass
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
    def random(cls, optimist_initial_value: float = 0) -> "OneArmedBandit":
        """
        Create a random one-arm bandit by sampling mu from N(0, 1) setting sigma = 1.0.

        Returns:
            OneArmedBandit: A random one-arm bandit.
        """
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
        self.bandits = [OneArmedBandit.random(optimist_initial_value) for _ in range(k)]
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

    def __epsilon_greedy(self, epsilon: float) -> tuple[float, bool]:
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


@dataclass
class ExperimentResult:
    epsilon: float
    """Epsilon value used in the experiment."""
    average_rewards: list[float]
    """Average rewards received across all simulations at a given action."""
    optimal_action_pct: list[float]
    """Percentage of optimal actions taken."""
    n: int = 0
    """Number of times the experiment has been run."""

    def update(self, action_rewards: list[float], optimal_actions: list[bool]) -> None:
        """
        Update the average rewards and optimal action percentage.

        Args:
            action_rewards (list[float]): List of rewards received across all simulations at a given action.
            optimal_actions (list[bool]): List of booleans indicating whether the action was optimal or not.

        Returns:
            None
        """
        if self.average_rewards is None:
            self.average_rewards = []
        if self.optimal_action_pct is None:
            self.optimal_action_pct = []

        self.n += 1
        self.average_rewards.append(np.mean(action_rewards))
        self.optimal_action_pct.append(np.mean(optimal_actions))


def run_experiments(
    k_armed_bandit: KArmedBanditExperiment,
    epsilons: list[float],
    n_actions: int,
    n_experiments: int,
) -> list[ExperimentResult]:
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
    results = [ExperimentResult(epsilon, [], [], 0) for epsilon in epsilons]

    for epslion in epsilons:
        for _ in range(n_experiments):
            # reset the bandits for each experiment
            k_armed_bandit.reset_bandits()

    # for experiment_idx in range(n_experiments):
    #     # reset the bandits for each experiment
    #     for bandit in bandits:
    #         bandit.n = 0
    #         bandit.q = 0.0
    #     for eps in epsilons:
    #         for _ in range(n_actions):
    #             step_reward = epsilon_greedy(bandits, eps)
    #             rewards[experiment_idx][eps].append(step_reward)

    # return rewards


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

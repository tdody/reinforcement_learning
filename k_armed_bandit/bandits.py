from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class OneArmedBandit:
    def __init__(
        self,
        mu: float,
        sigma: float,
        optimist_initial_value: float = 0,
        learning_rate: Optional[float] = None,
    ):
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
        self.learning_rate = learning_rate

    def pull(self) -> float:
        """
        Simulate pulling the arm of the bandit and update the estimated value of the bandit based on the observed reward.

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
        if self.learning_rate is None:
            alpha = 1 / self.n
        else:
            alpha = self.learning_rate

        # Update the estimated value of the bandit using incremental mean formula
        self.q = self.q + (reward - self.q) * alpha

        return reward

    def __repr__(self) -> str:
        """
        String representation of the OneArmedBandit.
        Returns:
            str: String representation of the OneArmedBandit.
        """
        return f"OneArmedBandit(mu={self.mu:.2f}, sigma={self.sigma:.2f}, n={self.n}, q={self.q:.2f}, learning_rate={self.learning_rate})"

    @classmethod
    def random_bandit(
        cls,
        optimist_initial_value: float = 0,
        random_seed: Optional[int] = None,
        learning_rate: Optional[float] = None,
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
        return cls(mu, sigma, optimist_initial_value, learning_rate)


class KArmedBanditExperiment:
    def __init__(
        self,
        k: int,
        epsilon: float,
        optimist_initial_value: float = 0,
        learning_rate: Optional[float] = None,
        random_seed: Optional[int] = None,
    ):
        """
        Initialize the K-armed bandit experiment with a given number of arms.
        Args:
            k (int): Number of armed bandits.
            epsilon (float): Probability of exploring.
            optimist_initial_value (float): Initial value of the bandit.
            learning_rate (float): Learning rate for updating the estimated value of the bandit.
            random_seed (int): Random seed for reproducibility.
        """

        self.k = k
        self.bandits = [
            OneArmedBandit.random_bandit(
                optimist_initial_value,
                learning_rate=learning_rate,
                random_seed=random_seed,
            )
            for _ in range(k)
        ]
        self.optimist_initial_value = optimist_initial_value
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.random_seed = random_seed

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

    def epsilon_greedy(self) -> tuple[float, bool]:
        """
        Choose a bandit using epsilon-greedy strategy.


        Returns:
            OneArmedBandit: The chosen bandit.
            boolean: True if the selected bandit is optimal, False otherwise.
        """
        selected_bandit: OneArmedBandit
        if np.random.rand() < self.epsilon:
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
    estimated_mus: dict[int, dict[str, float]]
    """Estimated mus for each bandit."""

    def __init__(self, epsilon: float, n_actions: int):
        self.epsilon = epsilon
        self.average_rewards = []
        self.optimal_action_pct = []
        self.n_experiments = 0
        self.n_actions = n_actions
        self.estimated_mus = {}

    def update(
        self,
        action_rewards: list[float],
        optimal_actions: list[bool],
        bandits: list[OneArmedBandit],
    ) -> None:
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

        # Update the estimated mus for each bandit
        for i, bandit in enumerate(bandits):
            


        # Normalize the average rewards and optimal action percentage
        self.average_rewards = [x / self.n_experiments for x in self.average_rewards]
        self.optimal_action_pct = [
            x / self.n_experiments for x in self.optimal_action_pct
        ]



def run_experiments(
    k: int,
    epsilons: list[float],
    n_actions: int,
    n_experiments: int,
    optimist_initial_value: float = 0,
    learning_rate: Optional[float] = None,
    random_seed: Optional[int] = None,
) -> dict[float, ExperimentResult]:
    """
    Run an experiment with a list of bandits using epsilon-greedy strategy.

    Args:
        k (int): Number of bandits.
        epsilons (list[float]): List of epsilon values for exploration.
        n_actions (int): Number of times to pull the bandit.
        n_experiments (int): Number of time we repeat teh experiment.
        optimist_initial_value (float): Initial value of the bandit.
        learning_rate (float): Learning rate for updating the estimated value of the bandit.
        random_seed (int): Random seed for reproducibility.

    Returns:
        list[ExperimentResult]: A list of ExperimentResult objects containing the
        epsilon value, average rewards, and optimal action percentage.
    """
    results = {epsilon: ExperimentResult(epsilon, n_actions) for epsilon in epsilons}

    for epsilon in epsilons:
        for _ in range(n_experiments):
            experiment = KArmedBanditExperiment(
                k,
                epsilon,
                optimist_initial_value,
                learning_rate,
                random_seed=random_seed,
            )
            step_rewards: list[float] = []
            optimal_actions: list[bool] = []

            for _ in range(n_actions):
                step_reward, is_optimal = experiment.epsilon_greedy()
                step_rewards.append(step_reward)
                optimal_actions.append(is_optimal)

            # update the average rewards and optimal action percentage
            results[epsilon].update(step_rewards, optimal_actions)

    return results


def plot_average_rewards(
    rewards: dict[float, ExperimentResult], ax: Optional[plt.Axes] = None
) -> None:
    """
    Plot the average rewards for each epsilon value.

    Args:
        rewards (list[dict[float, list[float]]]): A list of dictionaries where each

    Returns:
        None
    """

    if ax is None:
        fig, ax = plt.subplots()
    else:
        ax.clear()

    sns.set(style="white")
    for eps in rewards:
        ax.plot(
            rewards[eps].average_rewards,
            label=f"$\epsilon$={eps}",
        )
    ax.set_xlabel("Steps")
    ax.set_ylabel("Average Reward")
    n_experiments = len(rewards)
    ax.set_title(
        f"Epsilon-Greedy Strategy - Average Reward for {n_experiments} experiments"
    )
    ax.legend()

    if ax is None:
        plt.show()


def plot_optimal_action_pct(
    rewards: dict[float, ExperimentResult],
    ax: Optional[plt.Axes] = None,
) -> None:
    """
    Plot the optimal action percentage for each epsilon value.

    Args:
        rewards (list[dict[float, list[float]]]): A list of dictionaries where each
        dictionary contains the average rewards and optimal action percentage for a given epsilon value.

    Returns:
        None
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        ax.clear()

    sns.set(style="white")
    for eps in rewards:
        ax.plot(
            rewards[eps].optimal_action_pct,
            label=f"$\epsilon$={eps}",
        )
    ax.set_xlabel("Steps")
    ax.set_ylabel("Optimal Action Percentage")
    ax.set_title("Epsilon-Greedy Strategy - Optimal Action Percentage")
    # format the y-axis as a percentage
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.set_ylim(0, 1)
    ax.legend()

    if ax is None:
        plt.show()

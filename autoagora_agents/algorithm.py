# Copyright 2022-, Semiotic AI, Inc.
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod

import numpy as np
import torch
from torch import optim

import experiment
from autoagora_agents import buffer
from autoagora_agents.distribution import distributionfactory


class Algorithm(ABC):
    """Base class for algorithms.

    Concretions must implement :meth:`__call__`.

    Attributes:
        niterations (int): Number of times the algorithm has been called.
        nupdates (int): Number of times the algorithm has been updated.
        group (str): The group to which the algorithm belongs.
        i (int): The index of the algorithm.
        name (str): The group and index of the algorithm.
    """

    def __init__(self, *, group: str, i: int) -> None:
        self.niterations = 0
        self.nupdates = 0
        self.group = group
        self.i = i
        self.name = f"{group}_{i}"

    def reset(self) -> None:
        """Reset the algorithm's state."""
        self.niterations = 0

    def update(self) -> None:
        """Update the algorithm's parameters."""
        self.nupdates += 1

    @abstractmethod
    def __call__(
        self,
        *,
        observation: np.ndarray,
        action: np.ndarray,
        reward: float,
        done: bool,
    ) -> np.ndarray:
        """Run the algorithm forward.

        Keyword Arguments:
            observation (np.ndarray): The observation seen by the agent.
            action (np.ndarray): The previous action taken by the agent.
            reward (float): The reward of the agent.
            done (bool): If True, the agent is no longer in the game.

        Returns:
            np.ndarray: The next action taken by the agent.
        """
        pass

    @staticmethod
    def advantage(rewards: torch.Tensor) -> torch.Tensor:
        """Compute a simple advantage estimate.

        In effect, this is just standardising the samples to N(0, 1)

        Arguments:
            rewards (torch.Tensor): The reward-history using which to compute the advantage

        Returns:
            torch.Tensor: The advantage estimate
        """
        std = rewards.std()
        if torch.isnan(std) or std == 0:
            adv = rewards
        else:
            adv = (rewards - rewards.mean()) / rewards.std()
        return torch.unsqueeze(adv, dim=1)


class PredeterminedAlgorithm(Algorithm):
    """Change to a particular value at a given timestamp.

    Attributes:
        timestamps (list[int]): The timestamps at which to change the outputted value.
            Must start with 0.
        vals (list[np.ndarray]): The values outputted.
    """

    def __init__(
        self, *, group: str, i: int, timestamps: list[int], vals: list[np.ndarray]
    ) -> None:
        super().__init__(group=group, i=i)
        if timestamps[0] != 0:
            raise ValueError("The first timestamp must be 0.")
        if len(timestamps) != len(vals):
            raise ValueError("The timestamps and vals lists must have the same length")
        self.timestamps = timestamps
        self.vals = vals
        self.ix = 0

    def reset(self) -> None:
        super().reset()
        self.ix = 0

    def __call__(
        self,
        *,
        observation: np.ndarray,
        action: np.ndarray,
        reward: float,
        done: bool,
    ) -> np.ndarray:
        if self.ix != len(self.timestamps) - 1:
            if self.niterations >= self.timestamps[self.ix + 1]:
                self.ix += 1

        self.niterations += 1
        return self.vals[self.ix]


class BanditAlgorithm(Algorithm):
    """Algorithms that have no observation other than the reward.

    Keyword Arguments:
        group (str): The group to which the algorithm belongs.
        i (int): The id value of the object within the group.
        bufferlength (int): The length of the buffer storing historical samples.
        actiondistribution (dict): The config for the distribution representing the action.
        optimizer (dict): The config for the optimizer.

    Attributes:
        actiondist (Distribution): The distribution modelling action-selection.
        buffer (deque): The buffer storing historical samples.
        optimizer (optim.Optimizer): A torch optimizer.
    """

    def __init__(
        self,
        *,
        group: str,
        i: int,
        bufferlength: int,
        actiondistribution: dict,
        optimizer: dict,
    ) -> None:
        super().__init__(group=group, i=i)

        self.actiondist = distributionfactory(**actiondistribution)
        self.buffer = buffer.buffer(maxlength=bufferlength)
        optimizer["params"] = self.actiondist.params
        self.opt = optimizerfactory(**optimizer)

    def reset(self):
        super().reset()
        self.actiondist.reset()
        self.buffer.clear()

    def __call__(
        self,
        *,
        observation: np.ndarray,
        action: np.ndarray,
        reward: float,
        done: bool,
    ) -> np.ndarray:
        act = np.array(self.actiondist.sample())
        logprob = self.actiondist.logprob(torch.as_tensor(action))
        self.buffer.append(
            {
                "reward": reward,
                "action": action,
                "logprob": logprob,
            }
        )
        self.niterations += 1
        return act

    def logprob(self, actions: torch.Tensor) -> torch.Tensor:
        """Compute the log probability of the action given the distribution.

        Arguments:
            actions (torch.Tensor): The actions for which to compute the log probability

        Returns:
            torch.Tensor: The log probability of the actions.
        """
        return self.actiondist.logprob(actions)


# NOTE: This is experimental! Please do not use!
class VPGBandit(BanditAlgorithm):
    """Bandit using a Vanilla Policy Gradient update.

    Keyword Arguments:
        group (str): The group to which the algorithm belongs.
        i (int): The id value of the object within the group.
        bufferlength (int): The length of the buffer storing historical samples.
        actiondistribution (dict): The config for the distribution representing the action.
        optimizer (dict): The config for the optimizer.
    """

    def __init__(
        self,
        *,
        group: str,
        i: int,
        bufferlength: int,
        actiondistribution: dict,
        optimizer: dict,
    ) -> None:
        super().__init__(
            group=group,
            i=i,
            bufferlength=bufferlength,
            actiondistribution=actiondistribution,
            optimizer=optimizer,
        )

    def _vpgpiloss(self, *, reward: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Compute the VPG policy loss.

        Tries to push the policy to maximise the probability of taking actions that
        maximise the return via an advantage function, which is an lower-variance
        Q-function.

        Keyword Arguments:
            reward (torch.Tensor): The rewards associated with taking each action.
            action (torch.Tensor): The actions the agent took.

        Returns:
            torch.Tensor: The policy loss
        """
        adv = self.advantage(reward)
        logprob = self.logprob(action)

        # Treat the different gaussians as independent. Don't mean across them.
        loss = -torch.mean(logprob * adv, dim=0)
        return loss

    def update(self):
        if not buffer.isfull(self.buffer):
            return
        super().update()

        rewards = buffer.get("reward", self.buffer)
        actions = buffer.get("action", self.buffer)

        loss = self._vpgpiloss(reward=rewards, action=actions)

        # The fudge factor has been found to empirically be the best balance between the
        # standard deviation growing without exploding.
        fudgefactor = -5
        alexisterm = torch.exp(-self.actiondist.logstddev + fudgefactor)  # type: ignore
        loss += alexisterm

        # Backprop
        self.opt.zero_grad()
        torch.sum(loss).backward()
        self.opt.step()
        self.buffer.clear()


# NOTE: This is experimental. Do not use!
class PPOBandit(BanditAlgorithm):
    """Bandit with a PPO update.

    Keyword Arguments:
        group (str): The group to which the algorithm belongs.
        i (int): The id value of the object within the group.
        bufferlength (int): The length of the buffer storing historical samples.
        actiondistribution (dict): The config for the distribution representing the action.
        optimizer (dict): The config for the optimizer.
        ppoiterations (int): The number of iterations to update the policy for before
            stopping the update step.
        epsclip (float): The clip value.
        entropycoeff (float): How much to weight the entropy term in the loss.
        pullbackstrength (float): How strongly to apply pullback to the initial distribution.
        stddevfallback (bool): Whether to do fallback for the standard deviation.

    Attributes:
        ppoiterations (int): The number of iterations to update the policy for before
            stopping the update step.
        epsclip (float): The clip value.
        entropycoeff (float): How much to weight the entropy term in the loss.
        pullbackstrength (float): How strongly to apply pullback to the initial distribution.
        stddevfallback (bool): Whether to do fallback for the standard deviation.
    """

    def __init__(
        self,
        *,
        group: str,
        i: int,
        bufferlength: int,
        actiondistribution: dict,
        optimizer: dict,
        ppoiterations: int,
        epsclip: float,
        entropycoeff: float,
        pullbackstrength: float,
        stddevfallback: bool,
    ) -> None:
        super().__init__(
            group=group,
            i=i,
            bufferlength=bufferlength,
            actiondistribution=actiondistribution,
            optimizer=optimizer,
        )
        self.ppoiterations = ppoiterations
        self.epsclip = epsclip
        self.entropycoeff = entropycoeff
        self.pullbackstrength = pullbackstrength
        self.stddevfallback = stddevfallback

    def _ppoloss(
        self, *, actions: torch.Tensor, logprob: torch.Tensor, adv: torch.Tensor
    ) -> torch.Tensor:
        nlogprob = self.actiondist.logprob(actions)
        ratio = torch.exp(nlogprob - logprob)

        loss = -torch.min(
            ratio * adv,
            torch.clip(ratio, min=1 - self.epsclip, max=1 + self.epsclip) * adv,
        )
        return loss

    def _entropyloss(self) -> torch.Tensor:
        """Penalise high entropies."""
        return -self.actiondist.entropy() * self.entropycoeff

    def _update(self):
        if not buffer.isfull(self.buffer):
            return
        super().update()

        rewards = buffer.get("reward", self.buffer)
        actions = buffer.get("action", self.buffer)

        adv = self.advantage(rewards)
        logprob = self.logprob(actions).detach()

        for _ in range(self.ppoiterations):
            ppoloss = self._ppoloss(actions=actions, logprob=logprob, adv=adv)
            entropyloss = self._entropyloss()

            loss = torch.mean(ppoloss + entropyloss, dim=0)

            # Pullback
            loss += (
                torch.abs(self.actiondist.unclampedmean - self.actiondist.initial_mean)
                * self.pullbackstrength
            )

            if self.stddevfallback:
                diff = self.actiondist.logstddev - torch.log(
                    self.actiondist.initial_stddev
                )
                loss += torch.where(
                    diff > 0.0, diff * self.pullbackstrength, torch.zeros_like(diff)
                )

            self.opt.zero_grad()
            torch.sum(loss).backward()
            self.opt.step()

    def update(self):
        self._update()
        self.buffer.clear()


# NOTE: This is experimental. Do not use!
class RollingMemoryPPOBandit(PPOBandit):
    """Bandit with a PPO update wherein the buffer is maintained in an off-policy way.

    Keyword Arguments:
        group (str): The group to which the algorithm belongs.
        i (int): The id value of the object within the group.
        bufferlength (int): The length of the buffer storing historical samples.
        actiondistribution (dict): The config for the distribution representing the action.
        optimizer (dict): The config for the optimizer.
        ppoiterations (int): The number of iterations to update the policy for before
            stopping the update step.
        epsclip (float): The clip value.
        entropycoeff (float): How much to weight the entropy term in the loss.
        pullbackstrength (float): How strongly to apply pullback to the initial distribution.
        stddevfallback (bool): Whether to do fallback for the standard deviation.
    """

    def __init__(
        self,
        *,
        group: str,
        i: int,
        bufferlength: int,
        actiondistribution: dict,
        optimizer: dict,
        ppoiterations: int,
        epsclip: float,
        entropycoeff: float,
        pullbackstrength: float,
        stddevfallback: bool,
    ) -> None:
        super().__init__(
            group=group,
            i=i,
            bufferlength=bufferlength,
            actiondistribution=actiondistribution,
            optimizer=optimizer,
            ppoiterations=ppoiterations,
            epsclip=epsclip,
            entropycoeff=entropycoeff,
            pullbackstrength=pullbackstrength,
            stddevfallback=stddevfallback,
        )

    def logprob(self, _):
        return buffer.get("logprob", self.buffer).unsqueeze(dim=1)

    def update(self):
        self._update()


def algorithmgroupfactory(*, kind: str, count: int, **kwargs) -> list[Algorithm]:
    """Instantiate new algorithms for a particular group.

    Keyword Arguments:
        kind (str): The type of algorithm to instantiate.
            "vpgbandit" -> VPGBandit
            "ppobandit" -> PPOBandit
            "rmppobandit" -> RollingMemoryPPOBandit
            "predetermined" -> PredeterminedAlgorithm
        count (int): The number of entities in this group.

    Returns:
        list[Algorithm]: A list of instantiated algorithms.
    """
    algs = {
        "vpgbandit": VPGBandit,
        "ppobandit": PPOBandit,
        "rmppobandit": RollingMemoryPPOBandit,
        "predetermined": PredeterminedAlgorithm,
    }
    group = [experiment.factory(kind, algs, i=i, **kwargs) for i in range(count)]
    return group


def optimizerfactory(*, kind: str, **kwargs) -> optim.Optimizer:
    """Return the requested optimiser.

    Keyword Arguments:
        kind (str): The type of optimiser to instantiate.
            "adam" -> optim.Adam
            "sgd" -> optim.SGD
            "rmsprop" -> optim.RMSprop

    Returns:
        optim.Optimizer: The optimiser
    """
    opts = {"adam": optim.Adam, "sgd": optim.SGD, "rmsprop": optim.RMSprop}
    opt = experiment.factory(kind, opts, **kwargs)
    return opt

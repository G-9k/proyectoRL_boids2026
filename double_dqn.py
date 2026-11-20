"""
Double DQN (van Hasselt, Guez y Silver, 2016) sobre el DQN de Stable-Baselines3.

La única diferencia con DQN está en el objetivo de cada actualización:

    DQN:         y = r + γ · max_a' Q_target(s', a')
    Double DQN:  y = r + γ · Q_target(s', argmax_a' Q(s', a'))

En DQN, la misma red (la target) elige la mejor acción siguiente Y la evalúa. Como el max se queda
con la estimación más alta, tiende a quedarse con la que tiene el error más positivo: sobreestima.
En Double DQN, la red principal elige la acción y la red target la evalúa. Para sobreestimar, las
dos redes tendrían que equivocarse para arriba en la misma acción, lo que pasa mucho menos.

El resto del método train() es copia del de SB3 2.9.2 (stable_baselines3/dqn/dqn.py).
"""

import numpy as np
import torch as th
from torch.nn import functional as F
from stable_baselines3 import DQN


class DoubleDQN(DQN):
    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)

        losses = []
        for _ in range(gradient_steps):
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            discounts = replay_data.discounts if replay_data.discounts is not None else self.gamma

            with th.no_grad():
                # ---- Lo único distinto de DQN ----
                # 1) la red principal elige la mejor acción en s'
                mejor_accion = self.q_net(replay_data.next_observations).argmax(dim=1, keepdim=True)
                # 2) la red target dice cuánto vale esa acción
                next_q_values = th.gather(self.q_net_target(replay_data.next_observations), dim=1, index=mejor_accion)
                # ----------------------------------
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * discounts * next_q_values

            current_q_values = self.q_net(replay_data.observations)
            current_q_values = th.gather(current_q_values, dim=1, index=replay_data.actions.long())

            loss = F.smooth_l1_loss(current_q_values, target_q_values)
            losses.append(loss.item())

            self.policy.optimizer.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from ray.rllib.connectors.connector_v2 import ConnectorV2
from ray.rllib.core.columns import Columns
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import EpisodeType


class AddCostsFromInfos(ConnectorV2):
    """Adds a per-timestep `costs` column derived from infos.

    Expects the batch to already contain an `infos` column (Columns.INFOS) where each
    entry is a dict-like info per timestep.

    This connector will write a float32 array/tensor under key `"costs"`.
    Missing costs default to 0.0.
    """

    def __init__(
        self,
        input_observation_space=None,
        input_action_space=None,
        *,
        cost_info_key: str = "cost",
        output_key: str = "costs",
        default_cost: float = 0.0,
    ):
        super().__init__(input_observation_space, input_action_space)
        self.cost_info_key = cost_info_key
        self.output_key = output_key
        self.default_cost = default_cost

    @override(ConnectorV2)
    def __call__(
        self,
        *,
        episodes: List[EpisodeType],
        batch: Dict[str, Any],
        rl_module=None,
        **kwargs,
    ) -> Dict[str, Any]:
        # Batch is expected to be a dict: module_id -> module_batch.
        for module_id, module_batch in batch.items():
            if not isinstance(module_batch, dict):
                continue

            infos = module_batch.get(Columns.INFOS)
            if infos is None:
                continue

            # If already present, don't overwrite.
            if self.output_key in module_batch:
                continue

            costs = self._extract_costs_from_infos(infos)
            module_batch[self.output_key] = costs

        return batch

    def _extract_costs_from_infos(self, infos: Any) -> Any:
        # `infos` is typically a python list of dicts (not padded), or a list that may
        # include None entries. We keep numpy output here; downstream NumpyToTensor
        # (learner connector piece) will convert it.
        if isinstance(infos, np.ndarray):
            # If infos is an object array of dicts.
            infos_list = infos.tolist()
        else:
            infos_list = infos

        if not isinstance(infos_list, list):
            # Unexpected format; best-effort fallback.
            return np.array([], dtype=np.float32)

        costs: List[float] = []
        for info in infos_list:
            if isinstance(info, dict):
                v = info.get(self.cost_info_key, self.default_cost)
            else:
                v = self.default_cost
            try:
                costs.append(float(v))
            except (TypeError, ValueError):
                costs.append(float(self.default_cost))

        return np.asarray(costs, dtype=np.float32)

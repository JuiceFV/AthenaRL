from typing import Dict, List, Optional, Tuple, cast

import torch
from torch.nn import Module, Parameter

from athena.core.dtypes import Ftype
from athena.core.logger import LoggerMixin
from athena.core.parameters import NormalizationParams
from athena.preprocessing import MAX_FVALUE, MIN_FVALUE


class Preprocessor(Module, LoggerMixin):

    def __init__(
        self,
        normalization_params: Dict[int, NormalizationParams],
        device: Optional[torch.device] = None
    ) -> None:
        super().__init__()
        self.normalization_params = normalization_params
        self.fid2index, self.sorted_features, _ = self._sort_features_by_normalization()

        if device is not None:
            self.info(f"Using predefined device: {device.type}")
            self.device = device
        elif torch.cuda.is_available():
            self.info("Using GPU: Device wasn't directly passed and GPU available.")
            self.device = torch.device("cuda")
        else:
            self.info("Using CPU: Device wasn't directly passed and GPU not available.")
            self.device = torch.device("cpu")

        self.zero_tensor = Parameter(torch.tensor([0.0], device=self.device), requires_grad=False)
        self.one_tensor = Parameter(torch.tensor([1.0], device=self.device), requires_grad=False)
        self.negative_one_tensor = Parameter(torch.tensor([-1.0], device=self.device), requires_grad=False)
        self.one_hunderedth_tensor = Parameter(torch.tensor([0.01], device=self.device), requires_grad=False)
        self.min_tensor = Parameter(torch.tensor([-1e20], device=self.device), requires_grad=False)
        self.max_tensor = Parameter(torch.tensor([1e20], device=self.device), requires_grad=False)
        self.epsilon_tensor = Parameter(torch.tensor([1e-6], device=self.device), requires_grad=False)

        self.fheaders = self._get_feature_headers()
        self.split_sections: List[int] = []
        for i, ftype in enumerate(Ftype):
            begin_fheader, end_fheader = self._get_fheaders_iters(i)
            if begin_fheader == end_fheader:
                continue
            if ftype == Ftype.ENUM:
                for j in range(begin_fheader, end_fheader):
                    enum_norm_params = self.normalization_params[self.sorted_features[j]]
                    creator = getattr(self, "_create_params_" + ftype.value)
                    creator(j, enum_norm_params)
                    self.split_sections.append(1)
            else:
                norm_params = [
                    self.normalization_params[fid]
                    for fid in self.sorted_features[begin_fheader:end_fheader]
                ]
                creator = getattr(self, "_create_params_" + ftype.value)
                creator(begin_fheader, norm_params)
                self.split_sections.append(end_fheader - begin_fheader)

    def input_prototype(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.randn(1, len(self.normalization_params), device=self.device),
            torch.ones(1, len(self.normalization_params), dtype=torch.uint8, device=self.device)
        )

    def forward(self, input: torch.Tensor, input_presence: torch.Tensor) -> torch.Tensor:
        outputs = []
        split_input = torch.split(input, self.split_sections, dim=1)
        split_presence = torch.split(input_presence.float(), self.split_sections, dim=1)
        partition = 0
        for i, ftype in enumerate(Ftype):
            begin_fheader, end_fheader = self._get_fheaders_iters(i)
            if begin_fheader == end_fheader:
                continue
            if ftype == Ftype.ENUM:
                for j in range(begin_fheader, end_fheader):
                    norm_params = self.normalization_params[self.sorted_features[j]]
                    preprocessed_output = (
                        self._preprocess_feature(
                            j, split_input[partition], [norm_params]
                        ) * split_presence[partition]
                    )
                    partition += 1
                    self._check_preprocessed_output(preprocessed_output, [norm_params])
                    outputs.append(preprocessed_output)
            else:
                norm_params_list = [
                    self.normalization_params[fid]
                    for fid in self.sorted_features[begin_fheader:end_fheader]
                ]
                preprocessed_output = (
                    self._preprocess_feature(
                        begin_fheader, split_input[partition], norm_params_list
                    ) * split_presence[partition]
                )
                partition += 1
                self._check_preprocessed_output(preprocessed_output, norm_params_list)
                if ftype != Ftype.DO_NOT_PREPROCESS:
                    preprocessed_output = torch.clamp(preprocessed_output, MIN_FVALUE, MAX_FVALUE)
                outputs.append(preprocessed_output)

        return torch.cat(outputs, dim=1)

    def _preprocess_feature(
        self, begin_fheader: int, input: torch.Tensor, norm_params: List[NormalizationParams]
    ) -> torch.Tensor:
        ftype = norm_params[0].ftype
        preprocessor = getattr(self, "_preprocess_" + ftype.value)
        # FIXME: Further norm_params has to be unpacked for the ENUM
        return preprocessor(begin_fheader, input, norm_params)

    def _sort_features_by_normalization(self) -> Tuple[Dict[int, int], List[int], List[int]]:
        fid2index = {}
        sorted_features = []
        fheaders = []
        for ftype in Ftype:
            fheaders.append(len(sorted_features))
            for fid in sorted(self.normalization_params.keys()):
                norm = self.normalization_params[fid]
                if norm.ftype == ftype:
                    fid2index[fid] = len(sorted_features)
                    sorted_features.append(fid)
        return fid2index, sorted_features, fheaders

    def _get_feature_headers(self) -> List[int]:
        fheaders = []
        at_feature = -1
        for i, fid in enumerate(self.sorted_features):
            ftype = self.normalization_params[fid].ftype
            ftype_index = Ftype.item_index(ftype)
            if ftype_index < at_feature:
                raise IndexError("Sort features by their type, first")
            while ftype_index > at_feature:
                fheaders.append(i)
                at_feature += 1
        while at_feature < len(Ftype):
            fheaders.append(len(self.sorted_features))
            at_feature += 1
        return fheaders

    def _create_params_do_not_preprocess(self, begin_fheader: int, norm_params: List[NormalizationParams]):
        pass

    def _create_params_binary(self, begin_fheader: int, norm_params: List[NormalizationParams]):
        pass

    def _create_params_probability(self, begin_fheader: int, norm_params: List[NormalizationParams]):
        pass

    def _create_params_continuous(self, begin_fheader: int, norm_params: List[NormalizationParams]):
        self._create_param(begin_fheader, "means", torch.tensor([p.mean for p in norm_params], device=self.device))
        self._create_param(begin_fheader, "stdevs", torch.tensor([p.stdev for p in norm_params], device=self.device))

    def _create_params_boxcox(self, begin_fheader: int, norm_params: List[NormalizationParams]):
        self._create_param(
            begin_fheader,
            "shifts",
            torch.tensor([p.boxcox_shift for p in norm_params], device=self.device)
        )
        for p in norm_params:
            if abs(p.boxcox_lambda) <= 1e-6:
                raise ValueError(f"Invalid value for boxcox lambda: {str(p.boxcox_lambda)}")
        self._create_param(begin_fheader, "lambdas", torch.tensor(
            [p.boxcox_lambda for p in norm_params], device=self.device))
        self._create_params_continuous(begin_fheader, norm_params)

    def _create_params_quantile(self, begin_fheader: int, norm_params: List[NormalizationParams]):
        N = len(norm_params)
        nquantiles = torch.tensor(
            [[float(len(p.quantiles)) - 1 for p in norm_params]],
            device=self.device
        )
        self._create_param(begin_fheader, "nquantiles", nquantiles)

        max_num_quantile_boundaries = int(torch.max(torch.tensor([len(p.quantiles) for p in norm_params])).item())
        M = max_num_quantile_boundaries

        quantile_boundaries = torch.zeros([1, len(norm_params), max_num_quantile_boundaries], device=self.device)
        max_quantile_boundaries = torch.zeros([1, len(norm_params)], device=self.device)
        min_quantile_boundaries = torch.zeros([1, len(norm_params)], device=self.device)
        for i, p in enumerate(norm_params):
            quantile_boundaries[0, i, :] = p.quantiles[-1]
            quantile_boundaries[0, i, 0:len(p.quantiles)] = torch.tensor(p.quantiles, device=self.device)
            max_quantile_boundaries[0, i] = max(p.quantiles)
            min_quantile_boundaries[0, i] = min(p.quantiles)

        quantile_boundaries = quantile_boundaries.to(self.device)
        max_quantile_boundaries = max_quantile_boundaries.to(self.device)
        min_quantile_boundaries = min_quantile_boundaries.to(self.device)

        self._create_param(begin_fheader, "quantile_boundaries", quantile_boundaries)
        self._create_param(begin_fheader, "max_quantile_boundaries", max_quantile_boundaries)
        self._create_param(begin_fheader, "min_quantile_boundaries", min_quantile_boundaries)
        self._create_param(begin_fheader, "quantile_boundary_mask", torch.ones([1, N, M], device=self.device))

    def _create_params_enum(self, begin_fheader: int, norm_params: NormalizationParams):
        self._create_param(
            begin_fheader,
            "enum_values",
            torch.tensor(norm_params.possible_values, device=self.device, dtype=torch.float).unsqueeze(0)
        )

    def _preprocess_do_not_preprocess(
        self, begin_fheader: int, input: torch.Tensor, norm_params: List[NormalizationParams]
    ) -> torch.Tensor:
        return input

    def _preprocess_binary(
        self, begin_fheader: int, input: torch.Tensor, norm_params: List[NormalizationParams]
    ) -> torch.Tensor:
        return self.one_tensor - (input == self.zero_tensor).float()

    def _preprocess_probability(
        self, begin_fheader: int, input: torch.Tensor, norm_params: List[NormalizationParams]
    ) -> torch.Tensor:
        bounded_input = torch.clamp(input, 1e-5, 1 - 1e-5)
        return self.negative_one_tensor * (self.one_tensor / bounded_input - self.one_tensor).log()

    def _preprocess_continuous(
        self, begin_fheader: int, input: torch.Tensor, norm_params: List[NormalizationParams]
    ) -> torch.Tensor:
        means = self._fetch_param(begin_fheader, "means")
        stdevs = self._fetch_param(begin_fheader, "stdevs")
        return (input - means) / stdevs

    def _preprocess_boxcox(
        self, begin_fheader: int, input: torch.Tensor, norm_params: List[NormalizationParams]
    ) -> torch.Tensor:
        shifts = self._fetch_param(begin_fheader, "shifts")
        lambdas = self._fetch_param(begin_fheader, "lambdas")
        boxcox_output = (torch.pow(torch.clamp(input + shifts, 1e-6), lambdas) - self.one_tensor) / lambdas
        return self._preprocess_continuous(begin_fheader, boxcox_output, norm_params)

    def _preprocess_quantile(
        self, begin_fheader: int, input: torch.Tensor, norm_params: List[NormalizationParams]
    ) -> torch.Tensor:
        nquantiles = self._fetch_param(begin_fheader, "nquantiles")
        quantile_boundaries = self._fetch_param(begin_fheader, "quantile_boundaries")
        max_quantile_boundaries = self._fetch_param(begin_fheader, "max_quantile_boundaries")
        min_quantile_boundaries = self._fetch_param(begin_fheader, "min_quantile_boundaries")

        mask = self._fetch_param(begin_fheader, "quantile_boundary_mask")
        masked_inputs = input.unsqueeze(2) * mask

        input_geq = (masked_inputs >= quantile_boundaries).float()
        input_less = (masked_inputs < quantile_boundaries).float()
        max_clamp = (input >= max_quantile_boundaries).float()
        min_clamp = (input <= min_quantile_boundaries).float()
        min_or_max = (min_clamp + max_clamp).float()
        interpolate = (min_or_max < self.one_hunderedth_tensor).float()
        interpolate_left, _ = torch.max((input_geq * quantile_boundaries) + (input_less * self.min_tensor), dim=2)
        interpolate_right, _ = torch.min((input_less * quantile_boundaries) + (input_geq * self.max_tensor), dim=2)

        left_start = torch.sum(input_geq, dim=2) - self.one_tensor
        interpolated_values = (
            (
                left_start +
                (
                    (input - interpolate_left) /
                    (
                        (interpolate_right + self.epsilon_tensor) - interpolate_left
                    )
                )
            ) / nquantiles
        ).float()
        return max_clamp + (interpolate * interpolated_values).float()

    def _preprocess_enum(
        self, begin_fheader: int, input: torch.Tensor, norm_params: List[NormalizationParams]
    ) -> torch.Tensor:
        enum_values = self._fetch_param(begin_fheader, "enum_values")
        return (input == enum_values).float()

    def _create_param(self, begin_fheader: int, name: str, tensor: torch.Tensor) -> Parameter:
        param = Parameter(tensor, requires_grad=False)
        setattr(self, f"_auto_param_{str(begin_fheader)}_{name}", param)
        return param

    def _fetch_param(self, begin_fheader: int, name: str) -> Parameter:
        return cast(Parameter, getattr(self, f"_auto_param_{str(begin_fheader)}_{name}"))

    def _get_fheaders_iters(self, at: int) -> Tuple[int, int]:
        begin_fheader = self.fheaders[at]
        end_fheader = len(self.normalization_params) if (at + 1) == len(Ftype) else self.fheaders[at + 1]
        return begin_fheader, end_fheader

    def _check_preprocessed_output(self, batch: torch.Tensor, norm_params: List[NormalizationParams]):
        if not self.training:
            return
        ftype = norm_params[0].ftype
        min_value, max_value = batch.min(), batch.max()

        if ftype in (Ftype.BOXCOX, Ftype.CONTINUOUS, Ftype.DO_NOT_PREPROCESS):
            pass
        elif max_value.item() > MAX_FVALUE:
            raise ValueError(
                f"A {ftype.value} feature type upper bound is {max_value} which is greater "
                f"than empirically extracted {MAX_FVALUE}."
            )
        elif min_value.item() < MIN_FVALUE:
            raise ValueError(
                f"A {ftype.value} feature type lower bound is {min_value} which is less "
                f"than empirically extracted {MIN_FVALUE}."
            )

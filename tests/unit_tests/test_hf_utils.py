# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import importlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Protocol, cast
from unittest.mock import patch

from huggingface_hub import HfApi

from nemo_gym.config_types import DownloadJsonlDatasetHuggingFaceConfig
from nemo_gym.hf_utils import download_hf_dataset_as_jsonl


class _DatasetValue(Protocol):
    def to_json(self, path: str) -> object: ...


class _DatasetFactory(Protocol):
    @classmethod
    def from_list(cls, rows: list[dict[str, object]]) -> _DatasetValue: ...


class _DatasetDictValue(Protocol):
    def __getitem__(self, split: str) -> _DatasetValue: ...


class _DatasetDictFactory(Protocol):
    def __call__(self, splits: Mapping[str, _DatasetValue]) -> _DatasetDictValue: ...


class _DatasetsApi(Protocol):
    Dataset: _DatasetFactory
    DatasetDict: _DatasetDictFactory


datasets_api = cast(_DatasetsApi, importlib.import_module("datasets"))


class TestHFUtils:
    def test_sanity(self) -> None:
        HfApi()

    def test_download_structured_datasets_without_network(self, tmp_path: Path) -> None:
        train_rows: list[dict[str, object]] = [
            {"id": 1, "text": "train one"},
            {"id": 2, "text": "train two"},
            {"id": 3, "text": "train three"},
        ]
        validation_rows: list[dict[str, object]] = [
            {"id": 4, "text": "validation one"},
            {"id": 5, "text": "validation two"},
        ]
        dataset_dict = datasets_api.DatasetDict(
            {
                "train": datasets_api.Dataset.from_list(train_rows),
                "validation": datasets_api.Dataset.from_list(validation_rows),
            }
        )

        selected_dir = tmp_path / "selected"
        selected_config = DownloadJsonlDatasetHuggingFaceConfig.model_validate(
            {
                "repo_id": "org/dataset",
                "output_dirpath": str(selected_dir),
                "hf_token": "hf-token",
                "split": "validation",
            }
        )
        with patch("nemo_gym.hf_utils.load_dataset", return_value=dataset_dict["validation"]) as load_dataset:
            download_hf_dataset_as_jsonl(selected_config)

        load_dataset.assert_called_once_with("org/dataset", split="validation", token="hf-token")
        selected_lines = (selected_dir / "validation.jsonl").read_text(encoding="utf-8").splitlines()
        assert [json.loads(line) for line in selected_lines] == validation_rows

        all_dir = tmp_path / "all"
        all_config = DownloadJsonlDatasetHuggingFaceConfig.model_validate(
            {
                "repo_id": "org/dataset",
                "output_dirpath": str(all_dir),
                "hf_token": "hf-token",
            }
        )
        with patch("nemo_gym.hf_utils.load_dataset", return_value=dataset_dict) as load_dataset:
            download_hf_dataset_as_jsonl(all_config)

        load_dataset.assert_called_once_with("org/dataset", token="hf-token")
        assert sorted(path.name for path in all_dir.iterdir()) == ["train.jsonl", "validation.jsonl"]
        train_lines = (all_dir / "train.jsonl").read_text(encoding="utf-8").splitlines()
        validation_lines = (all_dir / "validation.jsonl").read_text(encoding="utf-8").splitlines()
        assert [json.loads(line) for line in train_lines] == train_rows
        assert [json.loads(line) for line in validation_lines] == validation_rows

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

from pathlib import Path

import pytest
import scripts.update_env_list as update_env_list


def test_main_updates_upstream_environment_document(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    destination = tmp_path / "UPSTREAM-ENVIRONMENTS.md"
    destination.write_text(
        "before\n<!-- START_TRAINING_SERVERS_TABLE -->old<!-- END_TRAINING_SERVERS_TABLE -->\nafter\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(update_env_list, "UPSTREAM_ENVIRONMENTS_PATH", destination)
    monkeypatch.setattr(update_env_list, "get_training_server_info", lambda: [])

    update_env_list.main()

    text = destination.read_text(encoding="utf-8")
    assert "before\n<!-- START_TRAINING_SERVERS_TABLE -->\n" in text
    assert "| Environment" in text
    assert "old" not in text
    assert "<!-- END_TRAINING_SERVERS_TABLE -->\n" in text


def test_main_reports_missing_upstream_environment_markers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    destination = tmp_path / "UPSTREAM-ENVIRONMENTS.md"
    destination.write_text("# Missing markers\n", encoding="utf-8")
    monkeypatch.setattr(update_env_list, "UPSTREAM_ENVIRONMENTS_PATH", destination)
    monkeypatch.setattr(update_env_list, "get_training_server_info", lambda: [])

    with pytest.raises(SystemExit, match="1"):
        update_env_list.main()

    assert "UPSTREAM-ENVIRONMENTS.md" in capsys.readouterr().err


def test_training_server_info_includes_benchmark_configs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    resources_servers = tmp_path / "resources_servers"
    responses_api_agents = tmp_path / "responses_api_agents"
    benchmarks = tmp_path / "benchmarks"
    resources_servers.mkdir()
    responses_api_agents.mkdir()
    benchmark_dir = benchmarks / "desktop_bench"
    benchmark_dir.mkdir(parents=True)
    agent_dir = responses_api_agents / "desktop_agent" / "configs"
    agent_dir.mkdir(parents=True)
    agent_config = agent_dir / "desktop_agent.yaml"
    agent_config.write_text(
        """
desktop_agent:
  responses_api_agents:
    desktop_agent:
      domain: agent
      description: Desktop benchmark
      value: GUI evaluation
""".lstrip(),
        encoding="utf-8",
    )
    (benchmark_dir / "config.yaml").write_text(
        f"""
config_paths:
  - {agent_config}

desktop_agent:
  responses_api_agents:
    desktop_agent:
      datasets:
        - name: validation
          type: validation
          license: Apache 2.0
          huggingface_identifier:
            repo_id: example/desktop-bench
""".lstrip(),
        encoding="utf-8",
    )

    monkeypatch.setattr(update_env_list, "RESOURCES_SERVERS_FOLDER", resources_servers)
    monkeypatch.setattr(update_env_list, "RESPONSES_API_AGENTS_FOLDER", responses_api_agents)
    monkeypatch.setattr(update_env_list, "BENCHMARKS_FOLDER", benchmarks)

    servers = update_env_list.get_training_server_info()

    assert len(servers) == 1
    server = servers[0]
    assert server.name == "desktop_bench"
    assert server.display_name == "Desktop Agent"
    assert server.config_path == "benchmarks/desktop_bench/config.yaml"
    assert server.readme_path == "benchmarks/desktop_bench/README.md"
    assert server.config_metadata.domain == "agent"
    assert server.config_metadata.description == "Desktop benchmark"
    assert server.config_metadata.value == "GUI evaluation"
    assert server.config_metadata.types == ["validation"]

# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Guard the published adapter dependency boundary."""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[2]


def load_pyproject(path: str) -> dict:
    return tomllib.loads(
        (ROOT_DIR / path / "pyproject.toml").read_text(encoding="utf-8")
    )


PACKAGE_VERSION = load_pyproject("")["project"]["version"]

ADAPTER_EXTRAS = {
    "claude": {
        "adapter": f"nemo-fabric-adapters-claude == {PACKAGE_VERSION}",
        "harness": ["claude-agent-sdk==0.2.120"],
    },
    "codex": {
        "adapter": f"nemo-fabric-adapters-codex == {PACKAGE_VERSION}",
        "harness": ["openai-codex==0.144.4"],
    },
    "deepagents": {
        "adapter": f"nemo-fabric-adapters-deepagents == {PACKAGE_VERSION}",
        "harness": [
            "deepagents>=0.6.12,<0.7.0",
            "langchain>=1.3,<2.0",
            "langgraph>=1.2,<2.0",
        ],
    },
    "hermes-agent": {
        "adapter": (
            f"nemo-fabric-adapters-hermes == {PACKAGE_VERSION}; "
            "python_version < '3.14'"
        ),
        "harness": ["hermes-agent>=0.17.0; python_version < '3.14'"],
    },
}


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("adapters/common", []),
        (
            "adapters/claude",
            [
                f"nemo-fabric-adapters-common == {PACKAGE_VERSION}",
                "tomli-w~=1.2",
            ],
        ),
        (
            "adapters/codex",
            [
                f"nemo-fabric-adapters-common == {PACKAGE_VERSION}",
                "tomli-w~=1.2",
            ],
        ),
        (
            "adapters/deepagents",
            [
                f"nemo-fabric-adapters-common == {PACKAGE_VERSION}",
                "langchain-mcp-adapters>=0.1,<0.3.0",
                "langchain-openai>=0.3",
                "langgraph-checkpoint-sqlite>=3.0,<4.0",
            ],
        ),
        (
            "adapters/hermes",
            [f"nemo-fabric-adapters-common == {PACKAGE_VERSION}"],
        ),
    ],
)
def test_adapter_runtime_dependencies(path: str, expected: list[str]):
    project = load_pyproject(path)["project"]
    assert project["version"] == PACKAGE_VERSION
    assert sorted(project.get("dependencies", [])) == sorted(expected)


def test_adapter_test_dependencies_are_root_only():
    manifest = load_pyproject("")
    expected = [
        dependency
        for extra in ADAPTER_EXTRAS.values()
        for dependency in extra["harness"]
    ]
    assert sorted(manifest["dependency-groups"]["adapter-tests"]) == sorted(expected)
    assert "adapter-tests" not in manifest["tool"]["uv"]["default-groups"]


@pytest.mark.parametrize("name", ADAPTER_EXTRAS)
def test_root_adapter_extras_split_minimal_and_complete_installs(name: str):
    extras = load_pyproject("")["project"]["optional-dependencies"]
    expected = ADAPTER_EXTRAS[name]

    assert extras[f"{name}-min"] == [expected["adapter"]]
    assert sorted(extras[name]) == sorted(
        [expected["adapter"], *expected["harness"]]
    )


def test_removed_hermes_extra_is_not_retained_as_an_alias():
    extras = load_pyproject("")["project"]["optional-dependencies"]
    assert "hermes" not in extras


def test_deepagents_relay_extra_does_not_install_the_harness():
    project = load_pyproject("adapters/deepagents")["project"]
    assert project["optional-dependencies"]["relay"] == ["nemo-relay>=0.6.0,<0.7"]

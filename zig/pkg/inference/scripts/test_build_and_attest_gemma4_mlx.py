from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from build_and_attest_gemma4_mlx import (
    ATTESTATION_NAME,
    BUILD_INPUTS_SCHEMA_VERSION,
    BUILD_LOG_NAME,
    RECEIPT_NAME,
    BuildInputs,
    BuildPaths,
    _attestation,
    _receipt,
    atomic_publish_json,
    build_command,
    dependency_tree_identity,
    discover_native_inventory,
    load_build_inputs,
    remove_generated_install_trees,
    require_closed_ignored_files,
    require_jaccl_inventory_sdk_version,
    resolve_bare_build_commands,
    resolve_build_paths,
    verify_pinned_source,
    verify_published_build,
)
from gemma4_oracle_contract import ContractError, LOCK_PATH, load_json, load_lock, prefixed_sha256


class MlxNativeBuildAttesterTest(unittest.TestCase):
    def make_dependency(self, root: Path, name: str) -> Path:
        path = (root / name).resolve()
        path.mkdir(parents=True)
        sentinels = {
            "fmt": ("CMakeLists.txt", "include/fmt/format.h"),
            "json": ("CMakeLists.txt", "single_include/nlohmann/json.hpp"),
            "metal_cpp": ("Metal/Metal.hpp",),
            "nanobind": ("CMakeLists.txt", "include/nanobind/nanobind.h"),
        }
        for relative in sentinels[name]:
            target = path / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(f"{name}:{relative}\n", encoding="utf-8")
        return path

    def make_tool(self, root: Path, name: str) -> Path:
        path = (root / name).resolve()
        path.write_bytes(b"\xcf\xfa\xed\xfe" + name.encode("ascii"))
        path.chmod(0o755)
        return path

    def make_inputs(self, root: Path) -> tuple[Path, BuildInputs]:
        dependency_root = (root / "dependencies").resolve()
        dependency_root.mkdir()
        dependencies = {
            name: self.make_dependency(dependency_root, name)
            for name in ("fmt", "json", "metal_cpp", "nanobind")
        }
        tool_root = (root / "tools").resolve()
        tool_root.mkdir()
        tool_paths = {
            "clang": self.make_tool(tool_root, "clang"),
            "clangxx": self.make_tool(tool_root, "clang++"),
            "cmake": self.make_tool(tool_root, "cmake"),
            "ninja": self.make_tool(tool_root, "ninja"),
            "python": Path(sys.executable).resolve(strict=True),
        }
        developer_dir = (root / "Developer").resolve()
        developer_dir.mkdir()
        manifest = {
            "schema_version": BUILD_INPUTS_SCHEMA_VERSION,
            "jobs": 4,
            "macos_deployment_target": "14.0",
            "macos_sdk_version": "26.2",
            "metal_toolchain_identifier": "com.apple.dt.toolchain.Metal.32023.864",
            "developer_dir": str(developer_dir),
            "user_home": str(Path.home().resolve()),
            "dependencies": {
                name: {
                    "path": str(path),
                    "tree_sha256": dependency_tree_identity(path)["tree_sha256"],
                }
                for name, path in dependencies.items()
            },
            "tools": {
                name: {"path": str(path), "sha256": prefixed_sha256(path)}
                for name, path in tool_paths.items()
            },
        }
        manifest_path = (root / "inputs.json").resolve()
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        return manifest_path, load_build_inputs(manifest_path)

    def make_native_package(self, root: Path) -> Path:
        package = root / "python" / "mlx"
        library = package / "lib"
        library.mkdir(parents=True)
        (package / "core.cpython-312-darwin.so").write_bytes(b"core")
        (library / "libjaccl.dylib").write_bytes(b"jaccl")
        (library / "libmlx.dylib").write_bytes(b"dylib")
        (library / "mlx.metallib").write_bytes(b"metal")
        return package

    def init_mlx_repo(self, root: Path, *, ignored: str = "build/\n") -> tuple[Path, str]:
        source = (root / "mlx").resolve()
        source.mkdir()
        (source / ".gitignore").write_text(ignored, encoding="utf-8")
        version = source / "mlx" / "version.h"
        version.parent.mkdir()
        version.write_text(
            "#define MLX_VERSION_MAJOR 0\n"
            "#define MLX_VERSION_MINOR 31\n"
            "#define MLX_VERSION_PATCH 2\n",
            encoding="utf-8",
        )
        (source / "python" / "mlx").mkdir(parents=True)
        subprocess.run(("git", "init", "-q", str(source)), check=True)
        subprocess.run(("git", "-C", str(source), "add", "."), check=True)
        subprocess.run(
            (
                "git",
                "-C",
                str(source),
                "-c",
                "user.name=Test",
                "-c",
                "user.email=test@example.invalid",
                "commit",
                "-q",
                "-m",
                "fixture",
            ),
            check=True,
        )
        revision = subprocess.run(
            ("git", "-C", str(source), "rev-parse", "HEAD"),
            check=True,
            text=True,
            stdout=subprocess.PIPE,
        ).stdout.strip()
        return source, revision

    def test_dependency_tree_is_closed_deterministic_and_ignores_git_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            source = root / "dep"
            source.mkdir()
            first = source / "a.txt"
            first.write_text("a", encoding="utf-8")
            nested = source / "nested" / "b.txt"
            nested.parent.mkdir()
            nested.write_text("b", encoding="utf-8")
            identity = dependency_tree_identity(source)
            self.assertEqual(identity, dependency_tree_identity(source))
            (source / ".git").mkdir()
            (source / ".git" / "noise").write_text("ignored", encoding="utf-8")
            self.assertEqual(identity, dependency_tree_identity(source))
            first.chmod(0o755)
            self.assertNotEqual(identity["tree_sha256"], dependency_tree_identity(source)["tree_sha256"])
            (source / "link").symlink_to(first)
            with self.assertRaisesRegex(ContractError, "symbolic link"):
                dependency_tree_identity(source)

    def test_build_inputs_are_strict_and_verify_explicit_tree_and_tool_pins(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest_path, inputs = self.make_inputs(Path(temporary).resolve())
            self.assertEqual(4, inputs.jobs)
            self.assertEqual(("fmt", "json", "metal_cpp", "nanobind"), tuple(pin.name for pin in inputs.dependencies))
            raw = load_json(manifest_path)
            raw["unknown"] = True
            manifest_path.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaisesRegex(ContractError, "unknown"):
                load_build_inputs(manifest_path)

    def test_closed_inventory_sdk_boundary_requires_jaccl_branch(self) -> None:
        for admitted in ("26.2", "26.2.0", "26.3", "27.0"):
            require_jaccl_inventory_sdk_version(admitted)
        for rejected in ("14.0", "15.5", "26.0", "26.1", "26.1.9"):
            with self.subTest(rejected=rejected):
                with self.assertRaisesRegex(ContractError, "JACCL"):
                    require_jaccl_inventory_sdk_version(rejected)

        with tempfile.TemporaryDirectory() as temporary:
            manifest_path, _inputs = self.make_inputs(Path(temporary).resolve())
            raw = load_json(manifest_path)
            raw["macos_sdk_version"] = "26.1"
            manifest_path.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaisesRegex(ContractError, "JACCL"):
                load_build_inputs(manifest_path)

    def test_build_command_is_fetch_disconnected_and_truthful_about_shell_and_network(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            _manifest_path, inputs = self.make_inputs(root)
            source = root / "mlx"
            output = source / "build" / "attested"
            paths = BuildPaths(
                source=source,
                output=output,
                package_root=source / "python" / "mlx",
                receipt=output / RECEIPT_NAME,
                attestation=output / ATTESTATION_NAME,
                log=output / BUILD_LOG_NAME,
            )
            toolchain = {"sdk_path": "/SDK"}
            source_identity = {"commit_unix_seconds": 123}
            command = build_command(paths, source_identity, inputs, toolchain)
            self.assertEqual("antfly_mlx_native_build_command/v1", command["schema_version"])
            self.assertFalse(command["network_policy"]["top_level_shell"])
            self.assertTrue(command["network_policy"]["upstream_shell_commands"])
            self.assertEqual("not_enforced", command["network_policy"]["network_isolation"])
            self.assertTrue(command["network_policy"]["fetchcontent_disconnected"])
            self.assertIn("-DFETCHCONTENT_FULLY_DISCONNECTED=ON", command["environment"]["CMAKE_ARGS"])
            self.assertIn("-DMLX_BUILD_PYTHON_STUBS=OFF", command["environment"]["CMAKE_ARGS"])
            self.assertIn("-DMLX_BUILD_GGUF=OFF", command["environment"]["CMAKE_ARGS"])
            self.assertIn(
                "-DCMAKE_CXX_FLAGS=--driver-mode=g++",
                command["environment"]["CMAKE_ARGS"],
            )
            self.assertEqual(
                "com.apple.dt.toolchain.Metal.32023.864",
                command["environment"]["TOOLCHAINS"],
            )
            self.assertIn("-DCMAKE_OSX_SYSROOT=/SDK", command["environment"]["CMAKE_ARGS"])
            self.assertEqual("1", command["environment"]["PIP_NO_INDEX"])
            self.assertNotIn("HTTP_PROXY", command["environment"])
            self.assertEqual("setup.py", command["argv"][1])
            self.assertIn("--inplace", command["argv"])

    def test_generated_install_cleanup_is_exact_ignored_and_rejects_symlinks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source, _revision = self.init_mlx_repo(
                Path(temporary).resolve(),
                ignored="build/\npython/mlx/include/\npython/mlx/share/\npython/mlx/lib/\n",
            )
            output = source / "build" / "attested"
            output.mkdir(parents=True)
            paths = BuildPaths(
                source=source,
                output=output,
                package_root=source / "python" / "mlx",
                receipt=output / RECEIPT_NAME,
                attestation=output / ATTESTATION_NAME,
                log=output / BUILD_LOG_NAME,
            )
            include = paths.package_root / "include"
            share = paths.package_root / "share"
            lib_cmake = paths.package_root / "lib" / "cmake"
            include.mkdir()
            share.mkdir()
            lib_cmake.mkdir(parents=True)
            (include / "mlx.h").write_text("header", encoding="utf-8")
            (share / "MLXConfig.cmake").write_text("config", encoding="utf-8")
            (lib_cmake / "jacclTargets.cmake").write_text("targets", encoding="utf-8")
            self.assertEqual(
                ["python/mlx/include", "python/mlx/share", "python/mlx/lib/cmake"],
                remove_generated_install_trees(paths),
            )
            self.assertFalse(include.exists())
            self.assertFalse(share.exists())
            self.assertFalse(lib_cmake.exists())

            include.mkdir()
            (include / "escape").symlink_to(paths.source / ".gitignore")
            with self.assertRaisesRegex(ContractError, "symbolic link"):
                remove_generated_install_trees(paths)

    def test_bare_command_resolution_rejects_a_shadowing_tool(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            expected_dir = root / "expected"
            shadow_dir = root / "shadow"
            expected_dir.mkdir()
            shadow_dir.mkdir()
            expected = self.make_tool(expected_dir, "cmake")
            shadow = self.make_tool(shadow_dir, "cmake")
            identity = resolve_bare_build_commands(str(expected_dir), {"cmake": expected})
            self.assertEqual(str(expected), identity["cmake"]["path"])
            with self.assertRaisesRegex(ContractError, "resolves to"):
                resolve_bare_build_commands(
                    f"{shadow_dir}:{expected_dir}",
                    {"cmake": expected},
                )

    def test_native_inventory_is_closed_and_uses_shared_canonical_digest(self) -> None:
        lock = load_lock(LOCK_PATH)
        with tempfile.TemporaryDirectory() as temporary:
            package = self.make_native_package(Path(temporary))
            inventory = discover_native_inventory(package, lock["mlx_reference"]["native_runtime"])
            self.assertEqual(
                ("jaccl-runtime-dylib", "metal-library", "python-extension", "runtime-dylib"),
                tuple(item["role"] for item in inventory["artifacts"]),
            )
            self.assertRegex(inventory["sha256"], r"^sha256:[0-9a-f]{64}$")
            (package / "lib" / "unexpected.dylib").write_bytes(b"extra")
            with self.assertRaisesRegex(ContractError, "closed four-artifact"):
                discover_native_inventory(package, lock["mlx_reference"]["native_runtime"])

    def test_native_inventory_rejects_symlinks_and_multiple_core_extensions(self) -> None:
        lock = load_lock(LOCK_PATH)
        with tempfile.TemporaryDirectory() as temporary:
            package = self.make_native_package(Path(temporary))
            (package / "core.other.so").write_bytes(b"other")
            with self.assertRaisesRegex(ContractError, "exactly one"):
                discover_native_inventory(package, lock["mlx_reference"]["native_runtime"])
        with tempfile.TemporaryDirectory() as temporary:
            package = self.make_native_package(Path(temporary))
            library = package / "lib"
            (library / "link.dylib").symlink_to(library / "libmlx.dylib")
            with self.assertRaisesRegex(ContractError, "symbolic link"):
                discover_native_inventory(package, lock["mlx_reference"]["native_runtime"])

    def test_source_must_be_exact_clean_root_and_output_must_be_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source, revision = self.init_mlx_repo(Path(temporary).resolve())
            lock = copy.deepcopy(load_lock(LOCK_PATH))
            lock["mlx_reference"]["source_revisions"]["mlx"] = revision
            identity = verify_pinned_source(source, lock)
            self.assertEqual(revision, identity["revision"])
            paths = resolve_build_paths(source, source / "build" / "attested", must_exist=False)
            self.assertEqual(source / "build" / "attested", paths.output)
            ignored_residue = source / "build" / "stale.o"
            ignored_residue.parent.mkdir()
            ignored_residue.write_bytes(b"stale")
            with self.assertRaisesRegex(ContractError, "ignored files"):
                require_closed_ignored_files(source)
            require_closed_ignored_files(source, build_output=source / "build")
            (source / "dirty.txt").write_text("dirty", encoding="utf-8")
            with self.assertRaisesRegex(ContractError, "clean"):
                verify_pinned_source(source, lock)
        with tempfile.TemporaryDirectory() as temporary:
            source, _revision = self.init_mlx_repo(Path(temporary).resolve(), ignored="")
            with self.assertRaisesRegex(ContractError, "gitignore"):
                resolve_build_paths(source, source / "build" / "attested", must_exist=False)

    def test_atomic_publication_is_validated_and_never_replaces(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "attestation.json"
            payload = {"schema_version": "test/v1", "value": 1}
            atomic_publish_json(path, payload)
            self.assertEqual(payload, load_json(path))
            before = path.read_bytes()
            with self.assertRaisesRegex(ContractError, "replace"):
                atomic_publish_json(path, {"schema_version": "test/v1", "value": 2})
            self.assertEqual(before, path.read_bytes())

    def test_atomic_publication_removes_target_when_directory_fsync_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "attestation.json"
            real_fsync = os.fsync
            calls = 0

            def fail_directory_fsync(descriptor: int) -> None:
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise OSError("injected directory fsync failure")
                real_fsync(descriptor)

            with mock.patch(
                "build_and_attest_gemma4_mlx.os.fsync",
                side_effect=fail_directory_fsync,
            ):
                with self.assertRaisesRegex(ContractError, "directory fsync failed"):
                    atomic_publish_json(path, {"schema_version": "test/v1"})
            self.assertFalse(path.exists())

    def test_post_write_verification_binds_receipt_runtime_lock_and_attestation(self) -> None:
        lock = load_lock(LOCK_PATH)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            _manifest_path, inputs = self.make_inputs(root)
            source = root / "mlx"
            package = self.make_native_package(source)
            output = source / "build" / "attested"
            output.mkdir(parents=True)
            paths = BuildPaths(
                source=source,
                output=output,
                package_root=package,
                receipt=output / RECEIPT_NAME,
                attestation=output / ATTESTATION_NAME,
                log=output / BUILD_LOG_NAME,
            )
            paths.log.write_text("build ok\n", encoding="utf-8")
            source_identity = {
                "path": str(source),
                "revision": lock["mlx_reference"]["source_revisions"]["mlx"],
                "version": lock["mlx_reference"]["packages"]["mlx"],
                "commit_unix_seconds": 123,
            }
            dependencies = [
                {"name": pin.name, **dependency_tree_identity(pin.path)}
                for pin in inputs.dependencies
            ]
            toolchain = {"sdk_path": "/SDK", "identity": "test"}
            command = build_command(paths, source_identity, inputs, toolchain)
            inventory = discover_native_inventory(package, lock["mlx_reference"]["native_runtime"])
            receipt = _receipt(
                lock_sha256=prefixed_sha256(LOCK_PATH),
                source_identity=source_identity,
                inputs=inputs,
                dependencies=dependencies,
                toolchain=toolchain,
                command=command,
                paths=paths,
                inventory=inventory,
                started_unix_ns=1,
                finished_unix_ns=2,
            )
            atomic_publish_json(paths.receipt, receipt)
            atomic_publish_json(paths.attestation, _attestation(lock, source_identity, inventory, paths.receipt))
            with mock.patch(
                "build_and_attest_gemma4_mlx.verify_pinned_source",
                return_value=source_identity,
            ), mock.patch("build_and_attest_gemma4_mlx.require_closed_ignored_files"):
                result = verify_published_build(
                    paths,
                    lock,
                    source_identity,
                    inputs,
                    dependencies,
                    toolchain,
                )
            self.assertTrue(result["ok"])
            (package / "lib" / "libmlx.dylib").chmod(0o644)
            (package / "lib" / "libmlx.dylib").write_bytes(b"tampered")
            with mock.patch(
                "build_and_attest_gemma4_mlx.verify_pinned_source",
                return_value=source_identity,
            ), mock.patch("build_and_attest_gemma4_mlx.require_closed_ignored_files"):
                with self.assertRaisesRegex(ContractError, "outputs"):
                    verify_published_build(
                        paths,
                        lock,
                        source_identity,
                        inputs,
                        dependencies,
                        toolchain,
                    )

    def test_attestation_shape_remains_runner_compatible(self) -> None:
        lock = load_lock(LOCK_PATH)
        with tempfile.TemporaryDirectory() as temporary:
            receipt = Path(temporary) / "receipt.json"
            receipt.write_text("{}\n", encoding="utf-8")
            source_identity = {"revision": lock["mlx_reference"]["source_revisions"]["mlx"]}
            inventory = {"sha256": "sha256:" + "a" * 64}
            attestation = _attestation(lock, source_identity, inventory, receipt)
            self.assertEqual(
                {
                    "schema_version",
                    "source_revision",
                    "source_clean",
                    "native_artifact_inventory_sha256",
                    "build_command_sha256",
                    "precision_policy_sha256",
                },
                set(attestation),
            )
            self.assertEqual(prefixed_sha256(receipt), attestation["build_command_sha256"])


if __name__ == "__main__":
    unittest.main()

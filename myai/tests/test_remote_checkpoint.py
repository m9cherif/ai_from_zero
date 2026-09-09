"""Tests for loading checkpoints by address rather than by path."""

import pytest

from ..checkpoint.remote import hf_to_url, is_remote, resolve_checkpoint


class TestIsRemote:
    @pytest.mark.parametrize("spec", [
        "https://example.com/model.pt",
        "http://example.com/model.pt",
        "hf://owner/repo/model.pt",
    ])
    def test_remote_specs(self, spec):
        assert is_remote(spec)

    @pytest.mark.parametrize("spec", [
        "output/checkpoints/checkpoint_latest.pt",
        "/absolute/path.pt",
        "./relative.pt",
    ])
    def test_local_paths(self, spec):
        assert not is_remote(spec)


class TestHfToUrl:
    def test_default_revision_is_main(self):
        assert hf_to_url("hf://owner/repo/model.pt") == \
            "https://huggingface.co/owner/repo/resolve/main/model.pt"

    def test_revision_is_honoured(self):
        assert hf_to_url("hf://owner/repo@v2/model.pt") == \
            "https://huggingface.co/owner/repo/resolve/v2/model.pt"

    def test_nested_path_is_preserved(self):
        assert hf_to_url("hf://owner/repo/sub/dir/model.pt") == \
            "https://huggingface.co/owner/repo/resolve/main/sub/dir/model.pt"

    @pytest.mark.parametrize("spec", ["hf://owner", "hf://owner/repo"])
    def test_malformed_spec_is_rejected(self, spec):
        with pytest.raises(ValueError, match="Malformed"):
            hf_to_url(spec)


class TestResolveCheckpoint:
    def test_local_path_is_returned_unchanged(self, tmp_path):
        """A local path must not be copied, hashed or otherwise touched."""
        target = tmp_path / "checkpoint.pt"
        target.write_bytes(b"weights")
        assert resolve_checkpoint(str(target)) == str(target)

    def test_missing_local_path_is_still_passed_through(self):
        # Reporting the missing file is torch.load's job, not the resolver's.
        assert resolve_checkpoint("/no/such/file.pt") == "/no/such/file.pt"

    def test_unreachable_url_raises_with_guidance(self, tmp_path):
        with pytest.raises(RuntimeError, match="HF_TOKEN"):
            resolve_checkpoint(
                "https://huggingface.co/does-not-exist-xyz/nope/resolve/main/a.pt",
                cache_dir=str(tmp_path),
            )

    def test_failed_download_leaves_no_partial_file(self, tmp_path):
        with pytest.raises(RuntimeError):
            resolve_checkpoint(
                "https://huggingface.co/does-not-exist-xyz/nope/resolve/main/a.pt",
                cache_dir=str(tmp_path),
            )
        # A half-written .part would be loaded as a valid cache hit next time.
        assert list(tmp_path.iterdir()) == []

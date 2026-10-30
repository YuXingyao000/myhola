"""CPU 回归：首视图读取、显式回退，以及完整 dataset 输出和 RNG 等价。"""

from __future__ import annotations

import io
import zipfile

import numpy as np
import pytest
import torch

from src.models.diffusion import image_loading


def write_npz_member(path, images, *, version=(1, 0), compression=zipfile.ZIP_DEFLATED):
    content = io.BytesIO()
    np.lib.format.write_array(content, images, version=version, allow_pickle=False)
    with zipfile.ZipFile(path, "w", compression=compression) as archive:
        archive.writestr("svr_imgs.npy", content.getvalue())


@pytest.mark.parametrize("version", [(1, 0), (2, 0)])
@pytest.mark.parametrize("compression", [zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED])
def test_first_view_is_exact_and_does_not_load_the_full_array(tmp_path, monkeypatch, version, compression):
    images = np.random.default_rng(17).integers(0, 256, (24, 31, 37, 3), dtype=np.uint8)
    path = tmp_path / "imgs.npz"
    write_npz_member(path, images, version=version, compression=compression)

    def reject_full_load(*args, **kwargs):
        raise AssertionError("the first-view fast path must not load the full array")

    monkeypatch.setattr(image_loading.np, "load", reject_full_load)
    actual = image_loading.load_image_view(path, "svr_imgs", 0, read_single_view=True)
    np.testing.assert_array_equal(actual, images[0])
    assert actual.flags.owndata and actual.flags.writeable
    assert actual.nbytes == images[0].nbytes


@pytest.mark.parametrize("rotation_id", [0, 7, 23])
def test_disabled_or_later_view_uses_existing_numpy_reader(tmp_path, monkeypatch, rotation_id):
    images = np.arange(24 * 7 * 9 * 3, dtype=np.uint8).reshape(24, 7, 9, 3)
    path = tmp_path / "imgs.npz"
    write_npz_member(path, images)

    def reject_first_view(*args, **kwargs):
        raise AssertionError("later views and the default path must use np.load")

    monkeypatch.setattr(image_loading, "_read_first_image", reject_first_view)
    actual = image_loading.load_image_view(path, "svr_imgs", rotation_id, read_single_view=rotation_id != 0)
    np.testing.assert_array_equal(actual, images[rotation_id])


@pytest.mark.parametrize("layout", ["fortran", "float32", "rank3", "npy_v3"])
def test_unsupported_layout_warns_and_preserves_numpy_result(tmp_path, layout):
    images = np.arange(24 * 7 * 9 * 3, dtype=np.uint8).reshape(24, 7, 9, 3)
    version = (1, 0)
    if layout == "fortran":
        images = np.asfortranarray(images)
    elif layout == "float32":
        images = images.astype(np.float32) / 255
    elif layout == "rank3":
        images = images[..., 0]
    elif layout == "npy_v3":
        version = (3, 0)
    path = tmp_path / "imgs.npz"
    write_npz_member(path, images, version=version)
    with pytest.warns(RuntimeWarning, match="falls back to np.load"):
        actual = image_loading.load_image_view(path, "svr_imgs", 0, read_single_view=True)
    np.testing.assert_array_equal(actual, images[0])


@pytest.mark.parametrize("corruption", ["header", "payload", "zip"])
def test_corrupt_input_is_not_silently_retried(tmp_path, monkeypatch, corruption):
    path = tmp_path / "imgs.npz"
    content = io.BytesIO()
    np.lib.format.write_array(content, np.zeros((24, 7, 9, 3), dtype=np.uint8), allow_pickle=False)
    data = content.getvalue()
    if corruption == "header":
        data = b"badNPY" + data[6:]
    elif corruption == "payload":
        data = data[:-1]
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("svr_imgs.npy", data)
    if corruption == "zip":
        path.write_bytes(path.read_bytes()[:-30])

    def reject_retry(*args, **kwargs):
        raise AssertionError("corrupt input must propagate its original read error")

    monkeypatch.setattr(image_loading.np, "load", reject_retry)
    with pytest.raises((ValueError, zipfile.BadZipFile)):
        image_loading.load_image_view(path, "svr_imgs", 0, read_single_view=True)


@pytest.fixture
def dataset_config(tmp_path):
    # 人工 fixture：没有读取训练之外的真实 evaluation 数据。
    generator = np.random.default_rng(43)
    split = tmp_path / "training.txt"
    split.write_text("one\ntwo\n")
    for model_id, num_faces in (("one", 3), ("two", 5)):
        image_dir = tmp_path / "conditions" / model_id
        image_dir.mkdir(parents=True)
        shape = (24, 17, 19, 3)
        np.savez_compressed(
            image_dir / "imgs.npz",
            svr_imgs=generator.integers(0, 256, shape, dtype=np.uint8),
        )
        np.savez_compressed(
            image_dir / "real_photo.npz", flux=generator.integers(0, 256, shape, dtype=np.uint8)
        )
        for rotation_id in range(24):
            cache_dir = tmp_path / "cache" / f"{model_id}_{rotation_id}"
            cache_dir.mkdir(parents=True)
            np.save(cache_dir / "features.npy", generator.standard_normal((num_faces, 64)).astype(np.float32))
    return dict(
        epoch_scale_factor=2, cached_latent_root=str(tmp_path / "cache"), raw_data_root=None,
        load_topology=False, train_dataset=str(split), val_dataset="NOT_READ", test_dataset="NOT_READ",
        padding="random", max_faces=7, is_aug=False, cached_condition=False, real_photo_ratio=0.0,
        condition_name="single_img", condition_data_root=str(tmp_path / "conditions"), is_overfit=False,
    )


def read_batch(dataset):
    np.random.seed(149)
    torch.manual_seed(149)
    batch = dataset.collate_fn([dataset[index] for index in (0, 1, 2)])
    return batch, np.random.get_state(), torch.random.get_rng_state()


def assert_same_batch(actual, expected, *, include_original):
    assert actual.keys() == expected.keys()
    for key in actual:
        if key == "conditions":
            expected_keys = expected[key].keys() if include_original else expected[key].keys() - {"ori_imgs"}
            assert actual[key].keys() == expected_keys
            for condition_key, tensor in actual[key].items():
                assert torch.equal(tensor, expected[key][condition_key]), condition_key
        elif isinstance(actual[key], torch.Tensor):
            assert torch.equal(actual[key], expected[key]), key
        else:
            assert actual[key] == expected[key], key


@pytest.mark.parametrize("condition_name", ["single_img"])
@pytest.mark.parametrize("padding", ["random", "zero"])
@pytest.mark.parametrize("is_aug", [False, True])
@pytest.mark.parametrize("real_photo_ratio", [0.0, 0.5])
def test_dataset_options_preserve_images_latents_padding_and_rng(
    dataset_config, condition_name, padding, is_aug, real_photo_ratio,
):
    from src.datasets.brep import Diffusion_dataset

    config = dict(dataset_config, condition_name=condition_name, padding=padding,
                  is_aug=is_aug, real_photo_ratio=real_photo_ratio)
    original = Diffusion_dataset("training", config)
    expected, numpy_state, torch_state = read_batch(original)
    assert "ori_imgs" in expected["conditions"]  # 缺省配置保持旧接口。
    for read_single_view, include_original in ((True, True), (False, False), (True, False)):
        dataset = Diffusion_dataset("training", dict(
            config, read_single_image_view=read_single_view, return_original_images=include_original,
        ))
        actual, actual_numpy_state, actual_torch_state = read_batch(dataset)
        assert_same_batch(actual, expected, include_original=include_original)
        assert actual_numpy_state[0] == numpy_state[0]
        np.testing.assert_array_equal(actual_numpy_state[1], numpy_state[1])
        assert actual_numpy_state[2:] == numpy_state[2:]
        assert torch.equal(actual_torch_state, torch_state)

"""The pair contact reader starts as a no-op, only moves the classifier, and never pulls the face decoder."""
import pytest
import torch

from src.models.vae import AutoEncoder_light
from src.models.vae.decoders.pair_contact import PairContactReader
from tests.test_intersection_vae import encoder_batch, tiny_config


def plain_and_reader(tmp_path):
    torch.manual_seed(3)
    plain = AutoEncoder_light(tiny_config()).eval()
    checkpoint = tmp_path / "legacy.ckpt"
    torch.save({"state_dict": {f"model.{k}": v for k, v in plain.state_dict().items()}}, checkpoint)
    model = AutoEncoder_light(tiny_config(checkpoint=str(checkpoint), pair_contact_reader=True)).eval()
    return plain, model


def decode(model, batch):
    encoding = model.encode(batch, True)
    z, _, _ = model.sample(encoding["face_features"], v_is_test=True)
    return model.decode({**encoding, "face_latents": z}, batch)


def test_contact_maps_read_distance_and_normals():
    grid = torch.stack(torch.meshgrid(torch.linspace(0, 1, 16), torch.linspace(0, 1, 16), indexing="ij"), -1)
    flat = torch.cat((grid, torch.zeros(16, 16, 1), torch.tensor([0., 0., 1.]).expand(16, 16, 3)), -1)
    shifted = flat.clone()
    shifted[..., 2] = 0.1
    maps = PairContactReader.contact_maps(flat[None], shifted[None])
    assert maps.shape == (1, 4, 16, 16)
    torch.testing.assert_close(maps[0, 0], torch.full((16, 16), 0.1), atol=1e-5, rtol=0)
    torch.testing.assert_close(maps[0, 1], torch.ones(16, 16), atol=1e-5, rtol=0)
    torch.testing.assert_close(maps[0, 2], torch.full((16, 16), 0.1), atol=1e-5, rtol=0)


def test_reader_starts_as_identity_and_only_changes_the_classifier(tmp_path):
    plain, model = plain_and_reader(tmp_path)
    assert model.pair_contact_reader is not None
    batch = encoder_batch()
    with torch.no_grad():
        plain_loss, plain_data = plain(batch, v_test=True)
        model_loss, model_data = model(batch, v_test=True)
        for key in plain_loss:
            torch.testing.assert_close(model_loss[key], plain_loss[key], rtol=0, atol=0)
        assert (model_data["pred_face_adj"] == plain_data["pred_face_adj"]).all()
        model.pair_contact_reader.out.weight.fill_(0.05)
        model.pair_contact_reader.out.bias.fill_(0.02)
        moved, reference = decode(model, batch), decode(plain, batch)
    assert not torch.equal(moved["loss_edge"], reference["loss_edge"])
    torch.testing.assert_close(moved["edge_points_local"], reference["edge_points_local"], rtol=0, atol=0)
    torch.testing.assert_close(moved["face_points_local"], reference["face_points_local"], rtol=0, atol=0)


def test_reader_gradient_never_reaches_the_face_decoder(tmp_path):
    _, model = plain_and_reader(tmp_path)
    model.train()
    with torch.no_grad():
        model.pair_contact_reader.out.weight.fill_(0.05)
    decode(model, encoder_batch())["loss_edge"].backward()
    assert model.pair_contact_reader.out.weight.grad is not None
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.pair_contact_reader.cnn.parameters())
    assert all(p.grad is None for p in model.face_points_decoder.parameters())
    assert all(p.grad is None for p in model.face_center_scale_decoder.parameters())


def test_checkpoint_loading_still_rejects_other_mismatches(tmp_path):
    plain, _ = plain_and_reader(tmp_path)
    state = {f"model.{k}": v for k, v in plain.state_dict().items()}
    del state["model.classifier.bias"]
    broken = tmp_path / "broken.ckpt"
    torch.save({"state_dict": state}, broken)
    with pytest.raises(RuntimeError, match="classifier.bias"):
        AutoEncoder_light(tiny_config(checkpoint=str(broken), pair_contact_reader=True))
    state["model.classifier.bias"] = plain.state_dict()["classifier.bias"]
    state["model.stray.weight"] = torch.zeros(1)
    torch.save({"state_dict": state}, broken)
    with pytest.raises(RuntimeError, match="stray"):
        AutoEncoder_light(tiny_config(checkpoint=str(broken)))

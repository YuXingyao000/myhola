import numpy as np
import torch

from src.preprocess.loop_sequences import canonical_loops, relations_from_events, EDGE, CLOSE, END, BOS, PAD
from src.models.vae.loop_blocks import event_grammar, OwnerBoundaryAR, MateDecoder, mate_objective, match_mates
from src.models.vae.decoders.intersection import AttnIntersection


def test_canonical_loops_survive_arbitrary_coedge_renumbering():
    owner = np.array([0, 0, 0, 0, 1, 1])
    nxt = np.array([1, 2, 0, 3, 5, 4])
    uv = np.random.default_rng(7).random((6, 16, 2))
    original = canonical_loops(owner, nxt, uv, 2)
    order = np.array([5, 1, 4, 2, 0, 3]); inverse = np.argsort(order)
    changed = canonical_loops(owner[order], inverse[nxt[order]], uv[order], 2)
    restored = [[[int(order[h]) for h in loop] for loop in face] for face in changed]
    assert restored == original


def test_events_support_holes_self_loops_and_do_not_close_truncations():
    sequences = [[EDGE, EDGE, CLOSE, EDGE, CLOSE, END], [EDGE, CLOSE, END]]
    owner, nxt, done = relations_from_events(sequences)
    assert owner.tolist() == [0, 0, 0, 1]
    assert nxt.tolist() == [1, 0, 2, 3] and done.all()
    owner, nxt, done = relations_from_events([[EDGE, EDGE]])
    assert nxt.tolist() == [-1, -1] and not done.any()
    assert not relations_from_events([[END]])[2].any()
    prefix = torch.tensor([[BOS, EDGE, EDGE, CLOSE, EDGE, CLOSE, END, PAD]])
    mask = event_grammar(prefix)[0]
    assert mask[0].tolist() == [True, False, False]
    assert mask[1].tolist() == [True, True, False]
    assert mask[3].tolist() == [True, False, True]
    assert mask[-1].tolist() == [False, False, True]


def test_ar_causality_and_teacher_shift_do_not_expose_target_event():
    torch.manual_seed(5)
    decoder = OwnerBoundaryAR(AttnIntersection(32, 32, 1, 64), 4, 8).eval()
    face = torch.randn(2, 32); memory = torch.randn(2, 16, 32)
    a = torch.tensor([[EDGE, EDGE, CLOSE, END], [EDGE, CLOSE, EDGE, CLOSE]])
    b = a.clone(); b[:, 2:] = torch.tensor([EDGE, CLOSE])
    ha, la = decoder.teacher_forcing(a, face, memory)
    hb, lb = decoder.teacher_forcing(b, face, memory)
    assert torch.allclose(ha[:, :3], hb[:, :3], atol=1e-6)
    assert torch.equal(la[:, :3], lb[:, :3])


def test_general_matching_is_reciprocal_allows_seams_and_keeps_null():
    # 0 and 1 may belong to the same owner; matching does not forbid seams.
    score = torch.full((5, 5), -5.)
    score.fill_diagonal_(-torch.inf)
    score[0, 1] = score[1, 0] = 4
    score[2, 3] = score[3, 2] = 3
    null = torch.zeros(5)
    mate = match_mates(score, null)
    assert mate.tolist() == [1, 0, 3, 2, -1]
    assert torch.equal(match_mates(score + 10, null + 10), mate)
    assert (match_mates(score - 10, null) == -1).all()


def test_ar_generation_stops_on_end_and_reports_limit_without_forced_close():
    decoder = OwnerBoundaryAR(AttnIntersection(32, 32, 1, 64), 4, 5).eval()
    face = torch.randn(2, 32); memory = torch.randn(2, 4, 32)
    with torch.no_grad():
        decoder.event_head.weight.zero_()
        decoder.event_head.bias.copy_(torch.tensor([0., 2., 3.]))
    events, _, _ = decoder.generate(face, memory)
    assert events.tolist() == [[EDGE, CLOSE, END]] * 2
    assert relations_from_events(events.numpy())[2].all()
    with torch.no_grad():
        decoder.event_head.bias.copy_(torch.tensor([3., 0., 0.]))
    events, _, _ = decoder.generate(face, memory)
    assert events.tolist() == [[EDGE] * 5] * 2
    assert not relations_from_events(events.numpy())[2].any()


def test_mate_symmetry_batch_mask_and_null_gradient():
    torch.manual_seed(2)
    model = MateDecoder(width=32, depth=1)
    h = torch.randn(4, 64, requires_grad=True); face = torch.randn(2, 32)
    owner = torch.tensor([0, 0, 1, 1]); objects = owner
    scores, null = model(h, face, owner, objects)
    assert torch.equal(scores, scores.T) and torch.isneginf(scores[:2, 2:]).all()
    # Keep only the first use: its deleted mate must become NULL.
    loss = mate_objective(scores, null, torch.tensor([1, 0, 3, 2]), 1., True)
    loss.backward()
    assert torch.isfinite(h.grad).all() and h.grad.abs().sum() > 0
    assert model.null.weight.grad.abs().sum() > 0


def test_loader_reorders_every_relation_and_preserves_packed_offsets(tmp_path):
    from src.datasets.loop_pcurve import LoopPcurveDataset
    from src.preprocess.native_pcurve import SCHEMA
    sample = tmp_path/'cad'; sample.mkdir()
    rng = np.random.default_rng(9)
    np.savez(sample/'data.npz', schema=SCHEMA,
             sample_points_faces=rng.normal(size=(3,16,16,6)).astype('float32'),
             sample_points_lines=rng.normal(size=(6,16,6)).astype('float32'),
             pcurve_uv=rng.random((6,16,2)).astype('float32'),
             owner=np.array([0,0,0,0,1,2]), next=np.array([1,2,3,0,4,5]),
             mate=np.array([4,3,5,1,0,2]), slot=np.zeros(6,dtype='int64'),
             side=np.array([0,0,0,1,1,1]), vertex=np.array([0,0,1,1,0,1]))
    split=tmp_path/'split.txt';split.write_text('cad\n')
    ds=LoopPcurveDataset('validation',dict(val_dataset=str(split),raw_data_root=str(tmp_path),is_aug=False,max_events=16))
    item=ds[0];batch=ds.collate_fn([item,item])
    assert torch.equal(batch['event_edge_ids'][batch['event_targets']==EDGE],torch.arange(12))
    assert torch.equal(batch['mate'][6:],batch['mate'][:6]+6)
    assert torch.equal(batch['next'][6:],batch['next'][:6]+6)
    owner,nxt,done=relations_from_events([x.tolist() for x in item['events']])
    assert np.array_equal(nxt,item['next'].numpy()) and done.all()
    assert np.array_equal(owner,item['owner'].numpy())
    assert (item['owner']==item['owner'][item['mate']]).sum()==2

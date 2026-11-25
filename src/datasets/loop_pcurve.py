"""Ordered loop events derived from native supervision, without changing UV."""
import torch
from torch.nn.utils.rnn import pad_sequence

from src.preprocess.loop_sequences import canonical_loops, EDGE, CLOSE, END
from .native_pcurve import NativePcurveDataset


class LoopPcurveDataset(NativePcurveDataset):
    def __init__(self, mode, cfg):
        super().__init__(mode, cfg)
        self.max_events = int(cfg['max_events'])

    def __getitem__(self, index):
        item = super().__getitem__(index)
        loops = canonical_loops(item['owner'].numpy(), item['next'].numpy(),
                                item['pcurve_uv'].numpy(), len(item['face_points']))
        order = torch.tensor([h for face in loops for loop in face for h in loop])
        inverse = torch.empty_like(order)
        inverse[order] = torch.arange(len(order))
        for key in ('edge_points', 'owner', 'pcurve_uv', 'slot', 'side', 'vertex'):
            item[key] = item[key][order]
        for key in ('next', 'mate'):
            item[key] = inverse[item[key][order]]
        sequences, event_edges, loop_ids, loop_owner = [], [], [], []
        for face, face_loops in enumerate(loops):
            events, edges = [], []
            for loop in face_loops:
                ids = inverse[loop]
                events.extend([EDGE] * len(loop) + [CLOSE])
                edges.extend(ids.tolist() + [-1])
                loop_ids.append(ids)
                loop_owner.append(face)
            events.append(END)
            edges.append(-1)
            if len(events) > self.max_events:
                raise ValueError(f"{item['v_prefix']}: {len(events)} events exceed capacity {self.max_events}")
            sequences.append(torch.tensor(events))
            event_edges.append(torch.tensor(edges))
        item.update(events=sequences, event_edges=event_edges, loop_ids=loop_ids,
                    loop_owner=torch.tensor(loop_owner))
        return item

    @staticmethod
    def collate_fn(items):
        batch = NativePcurveDataset.collate_fn(items)
        events, event_edges, loop_ids, loop_owner = [], [], [], []
        edge_offset = face_offset = 0
        for item in items:
            events.extend(item['events'])
            event_edges.extend(torch.where(ids >= 0, ids + edge_offset, -1) for ids in item['event_edges'])
            loop_ids.extend(ids + edge_offset for ids in item['loop_ids'])
            loop_owner.append(item['loop_owner'] + face_offset)
            edge_offset += len(item['edge_points'])
            face_offset += len(item['face_points'])
        batch.update(event_targets=pad_sequence(events, batch_first=True, padding_value=-100),
                     event_edge_ids=pad_sequence(event_edges, batch_first=True, padding_value=-1),
                     loop_edge_ids=pad_sequence(loop_ids, batch_first=True, padding_value=-1),
                     loop_owner=torch.cat(loop_owner))
        return batch

from collections import defaultdict
from tqdm import tqdm
from config import sim_config
import os
import re
import numpy as np
from dataclasses import dataclass
from argparse import ArgumentParser


@dataclass
class BsaBlockId:
    seq_id: int
    layer_id: int
    kvhead_id: int
    qhead_id: int
    block_id: int

    def __hash__(self):
        return hash(
            (self.seq_id, self.layer_id, self.kvhead_id, self.qhead_id, self.block_id)
        )

    def __eq__(self, other):
        if not isinstance(other, BsaBlockId):
            return False
        return (
            self.seq_id == other.seq_id
            and self.layer_id == other.layer_id
            and self.kvhead_id == other.kvhead_id
            and self.qhead_id == other.qhead_id
            and self.block_id == other.block_id
        )

    def __lt__(self, other):
        if not isinstance(other, BsaBlockId):
            return False
        return (
            self.seq_id,
            self.layer_id,
            self.kvhead_id,
            self.qhead_id,
            self.block_id,
        ) < (
            other.seq_id,
            other.layer_id,
            other.kvhead_id,
            other.qhead_id,
            other.block_id,
        )

    def __repr__(self):
        return f"BsaBlockId(seq_id={self.seq_id}, layer_id={self.layer_id}, kvhead_id={self.kvhead_id}, qhead_id={self.qhead_id}, block_id={self.block_id})"


class MobaTraceAnalyzer:
    def __init__(self):
        self.config = sim_config()
        self.traces = defaultdict(dict)
        self.access_log = defaultdict(list)
        self.block_stats = defaultdict(
            lambda: {"scores": [], "access_times": [], "access_count": 0}
        )

        self._parse_traces()

    def _parse_traces(self):
        all_files = set(os.listdir(self.config.trace_dir))

        # Extract unique request numbers
        trace_pattern = re.compile(r"trace(\d+)_layer(\d+)_(blocks|scores)\.npy")
        trace_files = defaultdict(lambda: defaultdict(dict))

        for filename in all_files:
            match = trace_pattern.match(filename)
            if match:
                seq_id = int(match.group(1))
                layer_idx = int(match.group(2))
                trace_type = match.group(3)
                assert trace_type in [
                    "blocks",
                    "scores",
                ], "Unknown trace type: {trace_type}. Should be blocks or scores"

                trace_files[seq_id][layer_idx][trace_type] = os.path.join(
                    self.config.trace_dir, filename
                )

        for seq_id in sorted(trace_files.keys()):
            for layer_idx in range(self.config.num_layers):
                if layer_idx not in trace_files[seq_id]:
                    raise ValueError(f"Layer {layer_idx} not found for seq {seq_id}")
                layer_data = trace_files[seq_id][layer_idx]
                if "blocks" not in layer_data:
                    raise ValueError(
                        f"Blocks not found for seq {seq_id} and layer {layer_idx}"
                    )
                if "scores" not in layer_data:
                    raise ValueError(
                        f"Scores not found for seq {seq_id} and layer {layer_idx}"
                    )

        num_traces = len(trace_files)
        for seq_id in tqdm(range(num_traces), desc="Loading traces"):
            for i in range(self.config.num_layers):
                block_trace = np.load(trace_files[seq_id][i]["blocks"])
                score_trace = np.load(trace_files[seq_id][i]["scores"])
                self.traces[seq_id][i] = (block_trace, score_trace)

        for seq_id in tqdm(range(len(trace_files)), desc="Indexing traces"):
            num_iters = self.traces[seq_id][0][0].shape[2]

            for iter_id in range(num_iters):
                for layer_id in range(self.config.num_layers):
                    block_ids = self.traces[seq_id][layer_id][0][:, :, iter_id]
                    scores = self.traces[seq_id][layer_id][1][:, :, iter_id]

                    # Process each query head
                    for qhead_id in range(self.config.num_heads):
                        kvhead_id = qhead_id // self.config.kv_group_size

                        qhead_blocks = block_ids[
                            :, qhead_id
                        ]  # topk blocks for this qhead
                        qhead_scores = scores[:, qhead_id]  # topk scores for this qhead

                        # Record each block access by this qhead
                        for topk_idx in range(len(qhead_blocks)):
                            block_id = qhead_blocks[topk_idx]
                            score = qhead_scores[block_id]

                            # Add to access log (preserves all qhead-level accesses)
                            bsa_block_id = BsaBlockId(
                                seq_id, layer_id, kvhead_id, qhead_id, block_id
                            )
                            self.access_log[bsa_block_id].append(
                                (iter_id, score.item(), topk_idx)
                            )

    # Get access logs for each block.
    # The log is a list of tuples (iter, score, topk_idx)
    def get_access_log(self):
        for bsa_block_id in sorted(self.access_log.keys()):
            access_log = self.access_log[bsa_block_id]
            print(bsa_block_id)
            access_log_rounded = [(item[0], round(item[1], 2), item[2]) for item in access_log]
            print(access_log_rounded)

    # Get the score distribution for each block.
    # Each head (seq_id, layer_id, kvhead_id, qhead_id) maps to the times each block
    # within that head is accessed, their average/variance scores.
    def analyze_score_distribution(self):
        access_freq_by_head = defaultdict(list)
        avg_score_by_head = defaultdict(list)
        var_score_by_head = defaultdict(list)
        for seq_id in range(len(self.traces)):
            for layer_id in range(self.config.num_layers):
                for kvhead_id in range(self.config.num_kvheads):
                    for qhead_id in range(
                        kvhead_id * self.config.kv_group_size,
                        (kvhead_id + 1) * self.config.kv_group_size,
                    ):
                        for block_id in range(self.config.num_blocks_per_head):
                            if (
                                seq_id,
                                layer_id,
                                kvhead_id,
                                qhead_id,
                            ) not in access_freq_by_head:
                                access_freq_by_head[
                                    (seq_id, layer_id, kvhead_id, qhead_id)
                                ] = [0] * self.config.num_blocks_per_head
                                avg_score_by_head[
                                    (seq_id, layer_id, kvhead_id, qhead_id)
                                ] = [0] * self.config.num_blocks_per_head
                                var_score_by_head[
                                    (seq_id, layer_id, kvhead_id, qhead_id)
                                ] = [0] * self.config.num_blocks_per_head
                            bsa_block_id = BsaBlockId(
                                seq_id, layer_id, kvhead_id, qhead_id, block_id
                            )
                            if bsa_block_id in self.access_log:
                                access_freq_by_head[
                                    (seq_id, layer_id, kvhead_id, qhead_id)
                                ][block_id] += len(self.access_log[bsa_block_id])
                                if len(self.access_log[bsa_block_id]) > 0:
                                    avg_score = sum(
                                        [
                                            access_log[1]
                                            for access_log in self.access_log[
                                                bsa_block_id
                                            ]
                                        ]
                                    ) / len(self.access_log[bsa_block_id])
                                    avg_score_by_head[
                                        (seq_id, layer_id, kvhead_id, qhead_id)
                                    ][block_id] += avg_score
                                    var_score_by_head[
                                        (seq_id, layer_id, kvhead_id, qhead_id)
                                    ][block_id] += sum(
                                        [
                                            (access_log[1] - avg_score) ** 2
                                            for access_log in self.access_log[
                                                bsa_block_id
                                            ]
                                        ]
                                    ) / len(
                                        self.access_log[bsa_block_id]
                                    )

        # for b in access_freq_by_head:
        #     avg_scores_rounded = [round(score, 2) for score in avg_score_by_head[b]]
        #     var_scores_rounded = [round(score, 2) for score in var_score_by_head[b]]
        #     print(f"Block {b}: access_freq={access_freq_by_head[b]}, avg_scores={avg_scores_rounded}, var_scores={var_scores_rounded}")

        print(self.traces[0][27][1][:, 15, 146])


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    config = sim_config()
    config.from_yaml(args.config)

    analyzer = MobaTraceAnalyzer()
    analyzer.get_access_log()
    # analyzer.analyze_score_distribution()


if __name__ == "__main__":
    main()

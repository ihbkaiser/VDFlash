import unittest

from specforge.data.utils import ExactShardSampler


class ExactShardSamplerTests(unittest.TestCase):
    def test_non_divisible_dataset_is_partitioned_exactly_once(self):
        shards = [
            list(ExactShardSampler(11, num_replicas=4, rank=rank))
            for rank in range(4)
        ]

        self.assertEqual([len(shard) for shard in shards], [3, 3, 3, 2])
        self.assertEqual([item for shard in shards for item in shard], list(range(11)))

    def test_more_ranks_than_rows_produces_empty_tail_shards(self):
        shards = [
            list(ExactShardSampler(2, num_replicas=4, rank=rank))
            for rank in range(4)
        ]

        self.assertEqual(shards, [[0], [1], [], []])

    def test_invalid_rank_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "rank"):
            ExactShardSampler(3, num_replicas=2, rank=2)


if __name__ == "__main__":
    unittest.main()

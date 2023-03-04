import time
from agq.core.structures import DataPoint, Naive, AGQ, AGQwithCM


class TestNaive:
    def test_merge(self, test_values):
        naive1 = Naive()
        naive2 = Naive()
        naive3 = Naive()
        for tagvalue, duration in test_values[:len(test_values)//2]:
            naive1.add(DataPoint(idx=tagvalue, value=duration))
            naive3.add(DataPoint(idx=tagvalue, value=duration))

        for tagvalue, duration in test_values[len(test_values)//2:]:
            naive2.add(DataPoint(idx=tagvalue, value=duration))
            naive3.add(DataPoint(idx=tagvalue, value=duration))

        naive1.merge(naive2)

        assert naive1.values_by_idx.keys() == naive3.values_by_idx.keys()
        for key in naive1.values_by_idx.keys():
            list_dp, list_dp_other = naive1.values_by_idx[key], naive3.values_by_idx[key]
            if not (sorted([e.value for e in list_dp]) == sorted([e.value for e in list_dp_other])):
                raise ValueError


class TestAGQ:
    def test_merge(self, test_values):
        num_digests = 10
        max_reservoir_size = 10
        agq1 = AGQ(max_digests=num_digests, max_reservoir_size=max_reservoir_size)
        agq2 = AGQ(max_digests=num_digests, max_reservoir_size=max_reservoir_size)
        agq3 = AGQ(max_digests=num_digests, max_reservoir_size=max_reservoir_size)
        naive = Naive()

        for tagvalue, duration in test_values[:len(test_values)//2]:
            dp = DataPoint(idx=tagvalue, value=duration, timestamp=time.time())
            agq1.add(dp)
            agq3.add(dp)
            naive.add(dp)

        for tagvalue, duration in test_values[len(test_values)//2:]:
            dp = DataPoint(idx=tagvalue, value=duration, timestamp=time.time())
            agq2.add(dp)
            agq3.add(dp)
            naive.add(dp)

        assert agq1.digests.keys() != agq3.digests.keys()
        assert agq2.digests.keys() != agq3.digests.keys()

        agq1.merge(agq2)

        assert agq1.points_processed == agq3.points_processed
        assert agq1.digests.keys() == agq3.digests.keys()

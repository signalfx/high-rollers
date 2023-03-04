from tdigest import TDigest

import numpy as np

from collections import Counter
from sortedcontainers import SortedSet
from typing import Optional, List, Dict
from pydantic import BaseModel

from probables import CountMinSketch

from . import probabilities


class AsymmetricTDigest(TDigest):

    def _threshold(self, q):
        if q <= 0.5:
            return self.n * self.delta
        else:
            return 4 * self.n * self.delta * q * (1 - q)


class DataPoint(BaseModel):
    idx: str
    value: float
    timestamp: Optional[float] = 1.0


class Naive(BaseModel):
    values_by_idx: Dict[str, List] = {}

    def add(self, y: DataPoint):
        if y.idx in self.values_by_idx.keys():
            self.values_by_idx[y.idx].append(y)
        else:
            self.values_by_idx[y.idx] = [y]

    def return_top(self, k: int = 100, p: int = 90) -> List[Dict]:
        _r = []
        for _k in self.values_by_idx.keys():
            _v = [y.value for y in self.values_by_idx[_k]]
            _p = np.percentile(_v, p)
            _r.append({'idx': _k, f'P{p}': _p, 'count': len(_v)})
        _r.sort(key=lambda x: - x[f'P{p}'])
        return _r[:k]

    def merge(self, other) -> None:
        if not isinstance(other, Naive):
            raise NotImplementedError

        for key, list_dp in other.values_by_idx.items():
            for dp in list_dp:
                self.add(dp)


class AGQEntry(BaseModel):
    idx: str
    digest: TDigest
    birth: Optional[float] = 1.0
    p90: Optional[float] = 0.0

    @classmethod
    def from_datapoint(cls, idx: str, y: DataPoint, excess_for_idx: int = 0):
        t = AsymmetricTDigest(delta=0.1)
        t.update(y.value)
        adjusted_90 = max(90 + (excess_for_idx / t.n) * (90 - 100), 0)
        p90 = t.percentile(p=adjusted_90)
        return cls(idx=idx, digest=t, birth=y.timestamp, p90=p90)

    class Config:
        arbitrary_types_allowed = True

    def percentile(self, p: float, count_to_insert: int = 0):
        adjusted_p = p + (count_to_insert / self.digest.n) * (p - 100)
        return self.digest.percentile(max(adjusted_p, 0))

    def count(self):
        return self.digest.n

    def add_datapoint(self, y: DataPoint, excess_for_idx: int = 0):
        self.digest.update(y.value)
        self.p90 = self.percentile(p=90, count_to_insert=excess_for_idx)


class BaseAGQ(BaseModel):
    points_processed = 0
    digests: Dict[str, AGQEntry] = {}
    p90_to_idx: SortedSet = SortedSet()
    max_digests: Optional[int] = 200

    class Config:
        arbitrary_types_allowed = True

    def _add_to_digests(self, y: DataPoint, excess_for_idx: int = 0):
        this_id = y.idx
        if this_id in self.digests.keys():  # self.monitored_idx:
            agq_entry = self.digests.get(this_id)
            self.p90_to_idx.remove((agq_entry.p90, this_id))
            agq_entry.add_datapoint(y, excess_for_idx=excess_for_idx)
            self.p90_to_idx.add((agq_entry.p90, this_id))
        elif len(self.digests) < self.max_digests:
            new_entry = AGQEntry.from_datapoint(this_id, y, excess_for_idx=excess_for_idx)
            self.digests[this_id] = new_entry  # AGQEntry(idx=this_id, digest=t, birth=y.timestamp, p90=p90)
            self.p90_to_idx.add((new_entry.p90, this_id))
        else:
            p90, idx = self.p90_to_idx[0]
            smallest = self.digests.get(idx)
            p = self._compute_replacement_probability(y, smallest)
            u = np.random.uniform(0, 1)
            if u <= p:
                self.p90_to_idx.remove((p90, idx))
                del self.digests[idx]
                new_entry = AGQEntry.from_datapoint(this_id, y, excess_for_idx=excess_for_idx)
                self.digests[this_id] = new_entry
                self.p90_to_idx.add((new_entry.p90, this_id))

    def _compute_replacement_probability(self, y: DataPoint, t_min: AGQEntry, **kwargs) -> float:
        return t_min.digest.cdf(y.value)

    def add(self, x: AGQEntry):
        pass

    def add(self, y: DataPoint, excess_for_idx: int = 0):
        self.points_processed += 1
        self._add_to_digests(y, excess_for_idx=excess_for_idx)

    def _merge_baseline_with_digests(self) -> List[AGQEntry]:
        pass

    def return_top(self, k: int = 100, p: int = 90) -> List[AGQEntry]:
        # assumed infrequent (or as "post-processing")
        l_ = self._merge_baseline_with_digests()
        l_.sort(key=lambda x: - x.percentile(p=p))
        return [{'idx': a.idx, 'P90': a.percentile(p=90), 'count': a.count()} for a in l_[:k]]

    def merge(self, other) -> None:
        if not isinstance(other, BaseAGQ):
            raise NotImplementedError

        self.max_digests = min(self.max_digests, other.max_digests)
        self.points_processed += other.points_processed

        for other_p90, other_key in other.p90_to_idx:
            if other_key in self.digests.keys():
                old_p90 = self.digests[other_key].p90

                # digests
                self.digests[other_key].digest += other.digests[other_key].digest
                self.digests[other_key].p90 = self.digests[other_key].percentile(p=90)
                self.digests[other_key].birth = min(self.digests[other_key].birth, other.digests[other_key].birth)

                # p90_to_idx
                self.p90_to_idx.remove((old_p90, other_key))
                self.p90_to_idx.add((self.digests[other_key].p90, other_key))

            else:
                # digests
                self.digests.update({other_key: other.digests[other_key]})

                # p90_to_idx
                self.p90_to_idx.add((other_p90, other_key))

        while len(self.p90_to_idx) > self.max_digests:
            min_p90, min_key = self.p90_to_idx.pop(0)
            self.digests.pop(min_key)


class AGQ(BaseAGQ):
    reservoir: Optional[List[DataPoint]] = []
    max_reservoir_size: Optional[int] = 1000

    class Config:
        arbitrary_types_allowed = True

    def _add_to_reservoir(self, y: DataPoint):
        if len(self.reservoir) <= self.max_reservoir_size:
            self.reservoir.append(y)
        else:
            u = np.random.uniform(0, 1)
            if u <= 1.0 / self.points_processed:
                self.reservoir[np.random.randint(self.max_reservoir_size)] = y

    def add(self, y: DataPoint):
        self._add_to_reservoir(y)
        super().add(y)

    def _merge_baseline_with_digests(self) -> List[AGQEntry]:
        n = Naive()
        for datapoint in self.reservoir:
            n.add(datapoint)
        l_ = self.digests.copy().values()
        weight = self.points_processed / max(len(self.reservoir), 1)
        for entry in l_:
            if entry.idx in n.values_by_idx.keys():
                for datapoint in n.values_by_idx[entry.idx]:
                    if datapoint.timestamp < entry.birth:
                        entry.digest.update(datapoint.value, w=weight)
        return list(l_)

    def merge(self, other) -> None:
        if not isinstance(other, AGQ):
            raise NotImplementedError

        super(AGQ, self).merge(other)

        assert self.max_reservoir_size == other.max_reservoir_size

        self.reservoir = []
        reservoir = sorted(self.reservoir + other.reservoir, key=lambda x: x.timestamp)
        for dp in reservoir:
            self._add_to_reservoir(dp)


class DummyAGQ(AGQ):
    def _compute_replacement_probability(self, y: DataPoint, t_min: AGQEntry, **kwargs) -> float:
        return 0.5


class SpaceSaving(BaseModel):
    capacity: int = 0
    ct_to_idx: SortedSet = SortedSet()
    counter: Counter = Counter()

    class Config:
        arbitrary_types_allowed = True

    def add(self, idx: str):
        if idx in self.counter or len(self.counter) < self.capacity:
            self.counter[idx] += 1
            if self.counter[idx] > 1:
                self.ct_to_idx.remove((self.counter[idx] - 1, idx))
            self.ct_to_idx.add((self.counter[idx], idx))
        else:
            # minimum
            _c, _idx = self.ct_to_idx[0]
            self.counter[idx] = _c
            del self.counter[_idx]
            self.ct_to_idx.remove((_c, _idx))
            self.ct_to_idx.add((_c, idx))

    def check(self, idx: str):
        return self.counter[idx]

    def merge(self, other) -> None:
        if not isinstance(other, SpaceSaving):
            raise NotImplementedError
        assert self.capacity == other.capacity
        self.counter += other.counter
        sorted = SortedSet((self.counter[i], i) for i in self.counter.keys())
        to_rm = len(sorted) - self.capacity
        for ct, idx in sorted[:to_rm]:
            del self.counter[idx]
        self.ct_to_idx = sorted[to_rm:]




class AGQwithSS(BaseAGQ):
    prob_method: str
    discontinuous: bool = False
    truncate_churn: bool = False
    space_saving: SpaceSaving = SpaceSaving()
    ss_capacity: int = 1000

    class Config:
        arbitrary_types_allowed = True
    def __init__(self, **kwargs):
        super(AGQwithSS, self).__init__(**kwargs)
        self.space_saving: SpaceSaving = SpaceSaving(capacity=self.ss_capacity)

    def add(self, y: DataPoint):
        self.space_saving.add(y.idx)
        ss_count = self.space_saving.check(y.idx)
        td_count = 0
        if y.idx in self.digests.keys():
            td_count = self.digests[y.idx].count()

        # cm has been updated but td has not
        ## here td_count could be larger anyway!!
        excess_count = max(ss_count - td_count - 1, 0)
        super().add(y, excess_for_idx=excess_count)

    def _compute_replacement_probability(self, y: DataPoint, t_min: AGQEntry,
                                         **kwargs) -> float:
        cdf = t_min.digest.cdf(y.value)
        candidate_total = self.space_saving.check(y.idx)
        incumbent_total, incumbent_digest = max(self.space_saving.check(t_min.idx), t_min.digest.n), t_min.digest.n
        if self.discontinuous is True:
            return self._discontinuous_formula(cdf, candidate_total, incumbent_total,
                                               incumbent_digest)
        else:
            if candidate_total <= 1:
                if cdf <= 0:
                    return 0.5 - incumbent_digest / 2 / incumbent_total
                else:
                    return 1. + (cdf - 1) * t_min.digest.n / incumbent_total
            else:
                if self.truncate_churn is False:
                    w_1 = candidate_total - 1
                    C = incumbent_total
                    T = incumbent_digest
                    # translate to "acceptance" probabilities
                    if self.prob_method == 'UNIFORM':
                        _p_1 = 1. - probabilities.UNIFORM[
                            int(max(min(w_1, len(probabilities.UNIFORM) - 1), 0))]
                    elif self.prob_method == 'UPDATING':
                        _p_1 = 1. - probabilities.UPDATING[
                            int(max(min(w_1, len(probabilities.UPDATING) - 1), 0))]
                    elif self.prob_method == 'INVERSE_WEIGHT':
                        _p_1 = 1. / (w_1 + 1.)
                    elif self.prob_method == 'CONSTANT':
                        _p_1 = 1 / 6.
                    else:
                        raise Exception
                    if cdf <= 0:
                        p_1 = _p_1 * T / C
                        p_2 = (w_1 + 1) * (C - T) / 2 / C
                    else:
                        p_1 = (_p_1 + cdf / w_1) * T / C
                        p_2 = (w_1 + 2) * (C - T) / 2 / C
                else:
                    w_1 = (candidate_total - 1) * incumbent_digest / incumbent_total
                    if self.prob_method == 'UNIFORM':
                        p_1 = 1. - probabilities.UNIFORM[
                            int(max(min(w_1, len(probabilities.UNIFORM) - 1), 0))]
                    elif self.prob_method == 'UPDATING':
                        p_1 = 1. - probabilities.UPDATING[
                            int(max(min(w_1, len(probabilities.UPDATING) - 1), 0))]
                    elif self.prob_method == 'INVERSE_WEIGHT':
                        p_1 = 1. / (w_1 + 1.)  # no solid justification for this one
                    elif self.prob_method == 'CONSTANT':
                        p_1 = 1 / 6.
                    else:
                        raise Exception
                    p_2 = cdf
                w_2 = 1.
                return (w_1 * p_1 + w_2 * p_2) / (w_1 + w_2)

    def _merge_baseline_with_digests(self) -> List[AGQEntry]:
        return list(self.digests.copy().values())
    def merge(self, other) -> None:
        if not isinstance(other, AGQwithSS):
            raise NotImplementedError

        self.max_digests = min(self.max_digests, other.max_digests)
        self.points_processed += other.points_processed

        self.space_saving.merge(other.space_saving)

        for other_p90, other_key in other.p90_to_idx:
            if other_key in self.digests.keys():
                old_p90 = self.digests[other_key].p90

                # digests
                self.digests[other_key].digest += other.digests[other_key].digest
                self.digests[other_key].p90 = self.digests[other_key].percentile(
                    p=90,
                    count_to_insert= max(self.space_saving.check(other_key) - self.digests[other_key].digest.n, 0)
                )
                self.digests[other_key].birth = min(self.digests[other_key].birth, other.digests[other_key].birth)

                # p90_to_idx
                self.p90_to_idx.remove((old_p90, other_key))
                self.p90_to_idx.add((self.digests[other_key].p90, other_key))

            else:
                # digests
                other.digests[other_key].p90 = other.digests[other_key].percentile(
                    p=90,
                    count_to_insert=max(self.space_saving.check(other_key) - other.digests[other_key].digest.n, 0)
                )
                self.digests.update({other_key: other.digests[other_key]})

                # p90_to_idx
                self.p90_to_idx.add((other.digests[other_key].p90, other_key))

        while len(self.p90_to_idx) > self.max_digests:
            min_p90, min_key = self.p90_to_idx.pop(0)
            self.digests.pop(min_key)

        assert self.discontinuous == other.discontinuous
        assert self.prob_method == other.prob_method
        assert self.truncate_churn == other.truncate_churn



class AGQwithCM(BaseAGQ):
    prob_method: str
    width: int = 1000
    depth: int = 20
    cm_sketch: CountMinSketch = CountMinSketch(width=width, depth=depth)
    discontinuous: bool = False
    truncate_churn: bool = False

    def __init__(self, **kwargs):
        super(AGQwithCM, self).__init__(**kwargs)
        self.cm_sketch: CountMinSketch = CountMinSketch(width=self.width, depth=self.depth)

    class Config:
        arbitrary_types_allowed = True

    def _add_to_cm_sketch(self, y: DataPoint):
        self.cm_sketch.add(y.idx)

    def add(self, y: DataPoint):
        self._add_to_cm_sketch(y)
        cm_count = self.cm_sketch.check(y.idx)
        td_count = 0
        if y.idx in self.digests.keys():
            td_count = self.digests[y.idx].count()

        # cm has been updated but td has not
        excess_count = max(cm_count - td_count - 1, 0)
        super().add(y, excess_for_idx=excess_count)

    def _compute_replacement_probability(self, y: DataPoint, t_min: AGQEntry, **kwargs) -> float:
        cdf = t_min.digest.cdf(y.value)
        candidate_total = self.cm_sketch.check(y.idx)
        incumbent_total, incumbent_digest = self.cm_sketch.check(t_min.idx), t_min.digest.n
        if self.discontinuous is True:
            return self._discontinuous_formula(cdf, candidate_total, incumbent_total, incumbent_digest)
        else:
            if candidate_total <= 1:
                if cdf <= 0:
                    return 0.5 - incumbent_digest / 2 / incumbent_total
                else:
                    return 1. + (cdf - 1) * t_min.digest.n / incumbent_total
            else:
                if self.truncate_churn is False:
                    w_1 = candidate_total - 1
                    C = incumbent_total
                    T = incumbent_digest
                    # translate to "acceptance" probabilities
                    if self.prob_method == 'UNIFORM':
                        _p_1 = 1. - probabilities.UNIFORM[
                            int(max(min(w_1, len(probabilities.UNIFORM) - 1), 0))]
                    elif self.prob_method == 'UPDATING':
                        _p_1 = 1. - probabilities.UPDATING[
                            int(max(min(w_1, len(probabilities.UPDATING) - 1), 0))]
                    elif self.prob_method == 'INVERSE_WEIGHT':
                        _p_1 = 1. / (w_1 + 1.)  # no solid justification for this one
                    elif self.prob_method == 'CONSTANT':
                        _p_1 = 1 / 6.
                    else:
                        raise Exception
                    if cdf <= 0:
                        p_1 = _p_1 * T / C
                        p_2 = (w_1 + 1) * (C - T) / 2 / C
                    else:
                        p_1 = (_p_1 + cdf / w_1) * T / C
                        p_2 = (w_1 + 2) * (C - T) / 2 / C
                else:
                    w_1 = (candidate_total - 1) * incumbent_digest / incumbent_total
                    if self.prob_method == 'UNIFORM':
                        p_1 = 1. - probabilities.UNIFORM[
                            int(max(min(w_1, len(probabilities.UNIFORM) - 1), 0))]
                    elif self.prob_method == 'UPDATING':
                        p_1 = 1. - probabilities.UPDATING[
                            int(max(min(w_1, len(probabilities.UPDATING) - 1), 0))]
                    elif self.prob_method == 'INVERSE_WEIGHT':
                        p_1 = 1. / (w_1 + 1.)  # no solid justification for this one
                    elif self.prob_method == 'CONSTANT':
                        p_1 = 1 / 6.
                    else:
                        raise Exception
                    p_2 = cdf
                w_2 = 1.
                return (w_1 * p_1 + w_2 * p_2) / (w_1 + w_2)

    @staticmethod
    def _discontinuous_formula(cdf, candidate_total, incumbent_total, incumbent_digest) -> float:
        if incumbent_total == incumbent_digest:  # no "excess" for incumbent
            if cdf > (1 - 2 ** max(- candidate_total, -20)):  # (2 ** candidate_total - 1) / (2 ** candidate_total):
                return 1.
        elif candidate_total == 1:  # excess for incumbent, but not for candidate
            adjusted_cdf = ((incumbent_total - incumbent_digest) + cdf * incumbent_digest) / incumbent_total
            if adjusted_cdf > 0.5:
                return 1.
        else:  # excess for both
            incumbent_fraction = incumbent_digest / incumbent_total
            adjusted_candidate_total = incumbent_fraction * candidate_total
            if cdf > (1 - 2 ** max(-adjusted_candidate_total,
                                   -20)):  # (2 ** adjusted_candidate_total - 1) / (2 ** adjusted_candidate_total):
                return 1.
        return 0.

    def _merge_baseline_with_digests(self) -> List[AGQEntry]:
        # count-min sketch has been kept in the loop throughout
        return list(self.digests.copy().values())

    def merge(self, other) -> None:
        if not isinstance(other, AGQwithCM):
            raise NotImplementedError

        self.max_digests = min(self.max_digests, other.max_digests)
        self.points_processed += other.points_processed

        self.cm_sketch.join(other.cm_sketch)

        for other_p90, other_key in other.p90_to_idx:
            if other_key in self.digests.keys():
                old_p90 = self.digests[other_key].p90

                # digests
                self.digests[other_key].digest += other.digests[other_key].digest
                self.digests[other_key].p90 = self.digests[other_key].percentile(
                    p=90,
                    count_to_insert=self.cm_sketch.check(other_key) - self.digests[other_key].digest.n
                )
                self.digests[other_key].birth = min(self.digests[other_key].birth, other.digests[other_key].birth)

                # p90_to_idx
                self.p90_to_idx.remove((old_p90, other_key))
                self.p90_to_idx.add((self.digests[other_key].p90, other_key))

            else:
                # digests
                other.digests[other_key].p90 = other.digests[other_key].percentile(
                    p=90,
                    count_to_insert=self.cm_sketch.check(other_key) - other.digests[other_key].digest.n
                )
                self.digests.update({other_key: other.digests[other_key]})

                # p90_to_idx
                self.p90_to_idx.add((other.digests[other_key].p90, other_key))

        while len(self.p90_to_idx) > self.max_digests:
            min_p90, min_key = self.p90_to_idx.pop(0)
            self.digests.pop(min_key)

        assert self.discontinuous == other.discontinuous
        assert self.prob_method == other.prob_method
        assert self.truncate_churn == other.truncate_churn

import time

from typing import Optional, List
from pydantic import BaseModel

from collections import Counter
from agq.core.structures import DataPoint, AGQ, AGQwithCM, Naive, BaseAGQ, DummyAGQ, AGQwithSS


class NaiveScenario(BaseModel):
    values: List[List]
    naive: Naive = None
    card: int = 0
    length: int = 0

    def _setup(self):
        self.naive = Naive()
        self.card, self.length = len(set([e[0] for e in self.values])), len(self.values)

    def run(self):
        self._setup()
        for e in self.values:
            dp = DataPoint(idx=e[0], value=e[1], timestamp=time.time())
            self.naive.add(dp)

    def summary(self):
            return f'''Tag has {self.card} values and {self.length} records, on average {self.length / self.card : 2f}'''


class Scenario(BaseModel):
    values: List[List]
    num_digests: Optional[int] = 200
    agq: BaseAGQ = None
    width: int = 1000
    depth: int = 20
    card: int = 0
    length: int = 0
    estimated: List[str] = None
    truth: List[str] = None
    analyzed: int = 0
    k: int = 0
    errors: Counter = Counter()
    detailed_errors: List[List] = []
    ss: bool = False
    ss_capacity : int = 1000

    # agq probability controls
    prob_method: str = 'UPDATING'
    truncate_churn: bool = False

    def _setup(self, dummy=False, cm=False, ss=False):
        if dummy + cm + ss >= 2:
            raise ValueError
        elif dummy is True:
            self.agq = DummyAGQ(max_digests=self.num_digests)
        elif ss is True:
            self.agq = AGQwithSS(max_digests=self.num_digests, prob_method=self.prob_method,
                                 truncate_churn=self.truncate_churn, ss_capacity=self.ss_capacity)
        elif cm is True:
            self.agq = AGQwithCM(max_digests=self.num_digests, prob_method=self.prob_method,
                                 truncate_churn=self.truncate_churn, width=self.width, depth=self.depth)
        else: # use naive, with reservoir
            self.agq = AGQ(max_digests=self.num_digests)
        self.card, self.length = len(set([e[0] for e in self.values])), len(self.values)

    def run(self, truncate=None, dummy=False, cm=False, ss=False, stdout=False):
        self._setup(dummy=dummy, cm=cm, ss=ss)
        n = 0
        self.analyzed = truncate if truncate is not None else self.length
        for e in self.values[:self.analyzed]:
            dp = DataPoint(idx=e[0], value=e[1], timestamp=time.time())
            self.agq.add(dp)
            n += 1
            if stdout:
                print(f'{n / self.analyzed * 100: 2f}% of records processed', end='\r')

    def summary(self):
        return f'''Tag has {self.card} values and {self.length} records, on average {self.length / self.card : 2f}; we analyzed {self.analyzed} and maintained (at most) 
        {self.num_digests} digests.'''

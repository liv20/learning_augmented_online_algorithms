from . import AbstractAlgorithm, OWTThresholdFunction

import numpy as np
from scipy.optimize import fsolve

class OneWayTradingAlgorithm(AbstractAlgorithm):
    def __init__(self, L, U, lmbda, seen_instances = [], predictor=None):
        """
        Initialize threshold function used to determine resource allocation at
        each time step. Starting resource amount (w) is set to 0.0, with 1.0 as
        the maximum. Can use predictions from predictor, which should implement
        the abstract_predictory.py class.
        """
        if predictor is None and lmbda < 1.0:
            raise ValueError("if no predictor, can only use lmbda == 1.0")

        # threshold function used in OneMaxSearch
        self.threshold = OWTThresholdFunction(L, U, lmbda)
        self.predictor = predictor
        self.seen_instances = seen_instances
        # resource utilization amount set to zero
        self.w = 0.0

    def allocate(self, instance):
        """
        Runs algorithm on an instance of data, allocating resources to maximize
        profit.
        Arguments:
        instance : pd.Series
            - array of time-series prices, for example, one week of BTC prices
        Returns: result (dictionary)
        - result['allocation'] : list showing allocation at each time step, should
            sum up to 1.0, the total amount allowed to be allocated
        - result['profit'] : profit generated by the algorithm
        """
        allocation = np.zeros(shape=(len(instance),), dtype=np.float32)
        # iterate over exchange rate in the time-series data
        for i in range(len(instance)):
            if self.predictor is not None:
                prediction = self.predictor.predict(self.seen_instances)
            else:
                # prediction will not be used in threshold calculation
                prediction = None
            reservation_price = self.threshold(self.w, prediction)

            if instance[i] < reservation_price:
                xn = 0
            elif instance[i] >= reservation_price and instance[i] < self.threshold(1, prediction):
                def func(x):
                    return self.threshold(self.w+x, prediction) - instance[i]
                xn = fsolve(func, [0.5])[0]

                # if proposed allocation is less than 0, set xn to 0
                if xn < 0:
                    xn = 0

                # if proposed allocation exceeds the remaining resources, allocate the remaining amount
                if xn > 1 - self.w:
                    xn = 1 - self.w
            else:
                xn = 1 - self.w
            # update allocation at ith timestep and the total resources allocated
            allocation[i] = xn
            self.w += xn
            if self.w >= 1:
                # break if all the resources have been allocated
                break
        # if there are remaining resources, exchange the remaining
        # resources at the last price
        if self.w <= 1:
            allocation[-1] += 1 - self.w

        result = {}
        result['allocation'] = allocation
        result['profit'] = np.sum(instance * allocation)

        return result
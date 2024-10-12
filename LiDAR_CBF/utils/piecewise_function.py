import torch
from triton.language import dtype


class DynamicPiecewiseFunction:
    def __init__(self, period):
        self.period = period
        self.funclist = []
        self.condlist = []
        self.condnum = 0

    def __call__(self, x):

        n2 = len(self.funclist)
        n = len(self.condlist)

        if n != n2:
            raise ValueError(
                f"With {n} condition(s), {n} functions are expected"
            )

        times = x[..., 0]
        out = torch.zeros_like(times)

        for condfn, func in zip(self.condlist, self.funclist):
            cond = condfn(times)
            if cond.any():
                out[cond] = func(x[cond].unsqueeze(1)).squeeze(1)

        return out

    def add_func(self, func):


        assert callable(func), 'Listed functions must be callable'
        self.condlist.append(self._generate_cond(self.condnum))
        self.funclist.append(func)
        self.condnum += 1

    def _generate_cond(self, k):
        """
        Generate a condition function for the k-th interval [k*T_s, (k+1)*T_s).
        """

        def cond(t):
            # return (t >= k * self.period) & (t <= (k + 1) * self.period)

            lower_bound = torch.tensor(k * self.period, dtype=torch.float64)
            upper_bound = torch.tensor((k + 1) * self.period, dtype=torch.float64)

            lower_check = torch.isclose(t, lower_bound, atol=1e-8) | (t > lower_bound)
            upper_check = torch.isclose(t, upper_bound, atol=1e-8) | (t < upper_bound)

            result = lower_check & upper_check
            return result

        return cond

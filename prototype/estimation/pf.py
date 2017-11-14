# from numpy import eye
# from numpy.linalg import inv


class PF(object):
    """ """

    def __init__(self):
        self.dt = None

        self.M = None
        self.R = None
        self.Q = None

        self.X = None
        self.Xp = None
        self.w = None
        self.mu = None
        self.S = None

    # def estimate(self, pf, Xp_func, hXp_func, u, y):
    #     # sampling
    #     for m = 1:pf.M
    #         Xp = feval(Xp_func, pf.X(m), u, pf.R);
    #         hXp = feval(hXp_func, Xp);
    #
    #         pf.Xp(:, m) = Xp;
    #         pf.w(m) = max(1e-8, mvnpdf(y, hXp, pf.Q));
    #     end
    #
    #     # importance resampling
    #     W = cumsum(pf.w);
    #     for m = 1:pf.M
    #         seed = W(end) * rand(1);
    #         pf.X(m) = pf.Xp(find(W > seed, 1));
    #     end
    #
    #     # record mean particle
    #     pf.mu = mean(pf.X);
    #     pf.S = var(pf.X);
    #
    # def localization_estimate(self, pf, Xp_func, hXp_func, mf, u, y):
    #     # sampling
    #     for m = 1:pf.M
    #         Xp = feval(Xp_func, u, R);
    #         hXp = feval(hXp_func, mf, Xp);
    #
    #         pf.Xp(:, m) = Xp;
    #         pf.w(m) = max(1e-8, mvnpdf(y, hXp, pf.Q));
    #     end
    #
    #     # importance resampling
    #     W = cumsum(pf.w);
    #     for m = 1:pf.M
    #         seed = W(end) * rand(1);
    #         pf.X(m) = pf.Xp(find(W > seed, 1));
    #     end
    #
    #     # record mean particle
    #     pf.mu = mean(pf.X);
    #     pf.S = var(pf.X);

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

import src.constants as constants

class CondScaler:
    def __init__(self, features, n_conditions=1, clip=10.0):
        self.features, self.n, self.clip = features, n_conditions, clip

    def _cond(self, df):
        if self.n == 1:
            return np.zeros(len(df), dtype=int)

        #cmapsdataset is separated into 6 regimes that vary slightly. separate entire dataset into 1 of the 6
        return self.km.predict(df[constants.operating_settings])

    def fit(self, df):
        if self.n > 1:
            self.km = KMeans(self.n, n_init=10, random_state=0).fit(df[constants.operating_settings])
        cond = self._cond(df)
        stds = df[self.features].groupby(cond).std()

        self.keep = [f for f in self.features if (stds[f] > 1e-6).all()]
        self.scalers = {c: StandardScaler().fit(df.loc[cond == c, self.keep]) for c in np.unique(cond)}
        return self

    def transform(self, df):
        out = df.copy()
        out[self.keep] = out[self.keep].astype("float64") 
        cond = self._cond(df)
        for c, sc in self.scalers.items():
            m = cond == c
            if m.any():
                out.loc[m, self.keep] = sc.transform(df.loc[m, self.keep])
        out[self.keep] = out[self.keep].clip(-self.clip, self.clip)
        return out
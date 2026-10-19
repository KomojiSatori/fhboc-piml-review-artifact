"""Compatible PIML readout with a secondary rule for exactly tied actions.

Primary objective: maximum posterior handoff-window mass (unchanged).
Secondary objective: maximum localization posterior at the window centre.
No near-tie tolerance, truth inputs, learned coefficients or new gates.
"""
import numpy as np
from scipy.special import logsumexp

from benchmark_fhboc_field_stop_20261016 import (
    OnlineBank, path_scores, state_features, angular_log_bf,
    standardized_features, complex_features, aggregate_features,
)


def select_window_action(mass, localization_log_posterior):
    mass=np.asarray(mass,float)
    ll=np.asarray(localization_log_posterior,float)
    assert mass.ndim==1 and mass.shape==ll.shape and len(mass)>0
    assert np.isfinite(mass).all() and np.all(mass>=0) and not np.isnan(ll).any()
    candidates=np.flatnonzero(mass==mass.max())
    return int(candidates[np.argmax(ll[candidates])])


class CenteredOnlineBank(OnlineBank):
    def readout(self,o,b,gate=None):
        E=self.d['observation_energy'][:,o];ps=path_scores(b,self.q,E,self.D);g=ps[:,2]
        packet={k:self.d[k] for k in ('cfg','axes','aiding','template_energy','states')}
        packet.update(projected=b[:,:,None],observation_energy=E[:,None],dimensions=self.D,path_scores=ps[:,None,:])
        x=standardized_features(state_features(packet,angular_log_bf(g[:,None],self.D.sum()))[0],self.model)
        bf=self.checkpoint.bf(g)
        base=self.prior+bf;ld=base+x@np.asarray(self.model['beta_detection'])
        cf=complex_features(b,self.q,E,self.D,self.d['states'],self.checkpoint.shape);phi=aggregate_features(cf,ld)
        score=float(logsumexp(ld)+np.clip((phi-self.model['complex_feature_mean'])/self.model['complex_feature_scale'],-4,4)@np.asarray(self.model['complex_feature_beta']))
        if gate is not None and score<=gate:return dict(score=score,detected=False)
        ll=base+x@np.asarray(self.model['beta_localization'])
        idx=select_window_action(self.plan.mass(np.exp(ll-ll.max())),ll)
        return dict(score=score,detected=gate is None or score>gate,fd_hat_hz=float(self.fd[idx]),tau_hat_samples=float(self.tau[idx]))

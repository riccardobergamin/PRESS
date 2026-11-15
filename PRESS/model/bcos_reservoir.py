"""BCOS transport with a protected waiting reservoir, retained explicitly."""
from dataclasses import asdict
import numpy as np
import jax.numpy as jnp
from scipy.special import expit

from .bcos import BCOSModel, _defender_times
from PRESS.physics.numerics import float64_context
from PRESS.physics.transport import RayMotion
from PRESS.physics.transport_reservoir import transport_probabilities


class ReservoirBCOSModel(BCOSModel):
    """Previous route-only interception; no killing before entry at the origin."""

    def predict(self,x_bc,v_bc,r_defs,v_defs,targets,*,time=np.inf):
        """Return route-level terminal/finite-time probabilities for (M,2) targets.

        XY is in centred pitch metres; velocities are m/s. Missing defenders
        use nonfinite positions or the existing >=500 m padding convention.
        A target equal to the current carrier position has accessibility one.
        """
        bc,velocity=np.asarray(x_bc,float).reshape(2),np.asarray(v_bc,float).reshape(2)
        if not np.isfinite(np.r_[bc,velocity]).all():
            raise ValueError('Carrier position/velocity must be finite.')
        defenders,def_velocity=np.asarray(r_defs,float),np.asarray(v_defs,float)
        targets=np.asarray(targets,float)
        if defenders.ndim!=2 or defenders.shape[1]!=2 or def_velocity.shape!=defenders.shape:
            raise ValueError('Defender positions and velocities must both have shape (D,2).')
        if targets.ndim!=2 or targets.shape[1]!=2 or not len(targets) or not np.isfinite(targets).all():
            raise ValueError('targets must have nonempty finite shape (M,2).')
        valid=np.isfinite(defenders).all(-1)&(np.linalg.norm(defenders,axis=-1)<500)
        if not np.isfinite(def_velocity[valid]).all():
            raise ValueError('Observed defenders require finite velocities.')
        active=defenders[valid]
        active_v=def_velocity[valid]
        cfg=self.config
        kappa=np.pi/(np.sqrt(3)*cfg.s)
        offset=targets-bc
        distance=np.linalg.norm(offset,axis=-1)
        origin=distance<=1e-12
        accessibility=np.ones(len(targets))
        survival=np.zeros(len(targets))
        reservoir=np.zeros(len(targets))
        interior=np.zeros(len(targets))
        interceptions=np.zeros((len(targets),len(defenders)))
        speed=float(np.linalg.norm(velocity))
        motion=RayMotion(speed,cfg.v_bc_max,cfg.a_bc_max)
        mass_error=0.
        with float64_context():
            for start in range(0,len(targets),cfg.batch_size):
                indices=np.arange(start,min(start+cfg.batch_size,len(targets)))
                indices=indices[~origin[indices]]
                if not len(indices):
                    continue
                unit=offset[indices]/distance[indices,None]
                if speed<.1:
                    angle=np.zeros(len(indices))
                else:
                    # Same target-angle convention as the shared TTI, held
                    # fixed along each straight ray to obtain regular tau(0).
                    angle=np.arccos(np.clip((unit@velocity)/speed,-1,1))
                delay=speed*angle/cfg.a_turn_bc
                duration=motion.travel_time(distance[indices])
                def rates(zeta,t):
                    if not len(active):
                        return np.zeros((*zeta.shape,0))
                    length=motion.distance(zeta)
                    points=bc+length[...,None]*unit[:,None,None,:]
                    tau=np.asarray(_defender_times(jnp.asarray(active),jnp.asarray(active_v),
                        jnp.asarray(points.reshape(-1,2)),cfg.v_def_max,cfg.a_def_max,cfg.a_turn_def))
                    tau=tau.T.reshape(*zeta.shape,len(active))
                    return kappa*expit(kappa*(t[...,None]-tau))
                result=transport_probabilities(duration,delay,cfg.s,rates,time=time,
                    arrival_order=cfg.arrival_order,path_order=cfg.path_order)
                accessibility[indices]=result['accessibility']
                survival[indices]=result['survival']
                reservoir[indices]=result['reservoir']
                interior[indices]=result['interior']
                interceptions[np.ix_(indices,np.flatnonzero(valid))]=result['interception']
                mass_error=max(mass_error,result['max_mass_error'])
        return {'Ball carrier open space':accessibility,'Defender interception probabilities':interceptions,
                'Defensive control':interceptions.sum(-1),'Unresolved probability':survival,
                'Reservoir probability':reservoir,'Interior probability':interior,
                'grid':targets.copy(),'max_mass_error':mass_error,
                'params':dict(formulation='transport_reservoir',waiting_interception=False,time=float(time),**asdict(cfg))}


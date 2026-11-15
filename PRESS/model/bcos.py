"""Ball-carrier accessibility from characteristic transport and interception."""
from dataclasses import asdict, dataclass

import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import expit

from PRESS.physics.times import time_to_intercept
from PRESS.physics.numerics import float64_context
from PRESS.physics.transport import RayMotion, transport_probabilities


@jax.jit
def _defender_times(positions, velocities, points, vmax, acceleration, turning):
    return time_to_intercept(positions,velocities,points,v_max=vmax,a_max=acceleration,
                             a_turn=turning,preserve_initial_overspeed=True)


@dataclass(frozen=True)
class BCOSConfig:
    """Physical scales and numerical orders; the dispersion comes from ball drive."""
    s: float = .3
    v_bc_max: float = 6.6
    v_def_max: float = 7.0
    a_bc_max: float = 7.0
    a_def_max: float = 7.0
    a_turn_bc: float = 7.0
    a_turn_def: float = 7.0
    arrival_order: int = 48
    path_order: int = 64
    batch_size: int = 8

    def __post_init__(self):
        for name in ['s','v_bc_max','v_def_max','a_bc_max','a_def_max','a_turn_bc','a_turn_def']:
            if not np.isfinite(getattr(self,name)) or getattr(self,name)<=0:
                raise ValueError(f'{name} must be finite and positive.')
        for name in ['arrival_order','path_order','batch_size']:
            if not isinstance(getattr(self,name),(int,np.integer)) or getattr(self,name)<(1 if name=='batch_size' else 2):
                raise ValueError(f'{name} has an invalid integer value.')


class BCOSModel:
    """Evaluate the final transport model without fitting a new predictor.

    The full initial law is conditioned on not being absorbed at time zero.
    Defender arrival hazards act during latent waiting at the origin and
    during subsequent movement along the route. Event probabilities partition into destination arrival,
    first interception by each defender, and survival (at finite time).
    """
    def __init__(self,config=None):
        self.config=config if isinstance(config,BCOSConfig) else BCOSConfig(**(config or {}))

    @classmethod
    def from_ball_drive(cls,model=None,**overrides):
        from .ball_drive import BallDriveModel
        model=model or BallDriveModel.pretrained()
        if isinstance(model,dict):
            model=BallDriveModel.from_parameters(model)
        elif not isinstance(model,BallDriveModel):
            model=BallDriveModel.from_json(model)
        physical=model.physics
        values=dict(s=float(np.exp(model.theta[3])),v_def_max=physical['v_max'],
                    a_def_max=physical['a_linear'],a_turn_def=physical['a_turn'])
        values.update(overrides)
        return cls(values)

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
            # Defender inputs are fixed for this frame; transfer them once.
            defender_positions = jnp.asarray(active)
            defender_velocities = jnp.asarray(active_v)
            # During latent waiting all paths occupy the same physical origin.
            # Arrival times there are constant; only the logistic time gate varies.
            origin_tau = (np.asarray(_defender_times(
                defender_positions, defender_velocities, jnp.asarray(bc[None]),
                cfg.v_def_max, cfg.a_def_max, cfg.a_turn_def,
            )).ravel() if len(active) else np.empty(0))
            def waiting_rates(t):
                return kappa*expit(kappa*(t[...,None]-origin_tau))
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
                    length=motion.distance(np.maximum(zeta,0))
                    points=bc+length[...,None]*unit[:,None,None,:]
                    tau=np.asarray(_defender_times(defender_positions,defender_velocities,
                        jnp.asarray(points.reshape(-1,2)),cfg.v_def_max,cfg.a_def_max,cfg.a_turn_def))
                    tau=tau.T.reshape(*zeta.shape,len(active))
                    return kappa*expit(kappa*(t[...,None]-tau))
                result=transport_probabilities(duration,delay,cfg.s,rates,time=time,
                    arrival_order=cfg.arrival_order,path_order=cfg.path_order,
                    waiting_rates=waiting_rates)
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
                'params':dict(formulation='transport',waiting_interception=True,time=float(time),**asdict(cfg))}

    def predict_field(self,x_bc,v_bc,r_defs,v_defs,*,r_max=40.,step=1.,pitch_length=105.,pitch_width=68.,time=np.inf):
        """Square-cell quadrature over the pitch-clipped disc centred on the carrier."""
        for name,value in dict(r_max=r_max,step=step,pitch_length=pitch_length,pitch_width=pitch_width).items():
            if not np.isfinite(value) or value<=0:
                raise ValueError(f'{name} must be finite and positive.')
        axis=np.arange(-r_max,r_max+1e-8,step)
        x,y=np.meshgrid(axis,axis)
        inside=x*x+y*y<=r_max*r_max+1e-9
        points=np.column_stack([x[inside],y[inside]])+np.asarray(x_bc)
        points=points[(abs(points[:,0])<=pitch_length/2)&(abs(points[:,1])<=pitch_width/2)]
        result=self.predict(x_bc,v_bc,r_defs,v_defs,points,time=time)
        result['BCOS area']=float(result['Ball carrier open space'].sum()*step**2)
        result['params'].update(r_max=r_max,step=step,pitch_length=pitch_length,pitch_width=pitch_width,cell_area=step**2)
        return result

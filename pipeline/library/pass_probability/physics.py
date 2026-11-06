"""Cache arrival times and hypothetical flight with frozen historical conventions."""
from functools import partial
import jax
import jax.numpy as jnp
import numpy as np
from PRESS.physics.flight_ball import generate_ball_trajectories,time_flight
from PRESS.physics.times import time_to_intercept
from .precision import enable_x64

# Same flight constants as the historical pass-feature producer, not a new fit.
FLIGHT=dict(t_max_flight=10.,dt_ball=.04,grid_step=1.,z_max=2.,rho=1.22,Cd=.25,A=.038,m=.42,g=9.81)


def candidates():
    return jnp.linspace(5.,25.,30,dtype=jnp.float32),jnp.linspace(0.,jnp.pi/4,30,dtype=jnp.float32)


TTI_SETTINGS=dict(a_turn=7.,preserve_initial_overspeed=True)


@partial(jax.jit,static_argnames=('vmax','a_turn','preserve_initial_overspeed'))
def arrival_batch(pos,vel,target,*,vmax,a_turn=None,preserve_initial_overspeed=False):
    return jax.vmap(lambda p,v,r:time_to_intercept(p,v,r[None],v_max=vmax,a_max=7.,a_turn=a_turn,
        preserve_initial_overspeed=preserve_initial_overspeed).squeeze(-1))(pos,vel,target)


def prepare_arrays(raw,vmax):
    # Arrival arithmetic is deliberately benchmark float32, fitting is float64.
    with enable_x64(False):
        pos,vel,target,bc=[jnp.asarray(raw[k],dtype=jnp.float32) for k in ['p_pos','p_vel','r_target','x_bc']]
        tau=arrival_batch(pos,vel,target,vmax=vmax)
        valid=np.linalg.norm(np.asarray(pos),axis=-1)<500;valid[:,10]=False
        time=jnp.arange(0.,10.+.04,.04,dtype=jnp.float32)
        speeds,angles=candidates()
        trajectory=generate_ball_trajectories(speeds,angles,time,rho=FLIGHT['rho'],Cd=FLIGHT['Cd'],A=FLIGHT['A'],m=FLIGHT['m'],g=FLIGHT['g'])
        distances=jnp.linalg.norm(target-bc,axis=-1)
        # Preserve the existing SINGLE-target linspace/index convention exactly.
        count=(distances/FLIGHT['grid_step']).astype(jnp.int32)+2
        index=jnp.clip(jnp.rint(distances/FLIGHT['grid_step']).astype(jnp.int32),0,count-1)
        effective_radius=distances*index/(count-1)
        nearest=jnp.argmin(jnp.linalg.norm(pos[:,:10]-target[:,None,:],axis=-1),axis=1)
        player_time=jnp.take_along_axis(tau[:,:10],nearest[:,None],axis=1)[:,0]
        @jax.jit
        def choose(radius,desired):
            def one(r,t):
                times=jax.vmap(lambda tr:jnp.interp(r,tr[:,0],time))(trajectory)
                heights=jax.vmap(lambda tr,hit:jnp.interp(hit,time,tr[:,1]))(trajectory,times)
                available=(heights<=2.) & (r<=trajectory[:,:,0].max(axis=1))
                times=jnp.where(available,times,jnp.inf)
                return times[jnp.argmin(jnp.abs(times-t))]
            return jax.vmap(one)(radius,desired)
        t_ball=np.asarray(choose(effective_radius,player_time))
        result=dict(tau=np.asarray(tau),t_ball=t_ball,valid=valid)
        checks=[]
        for i in np.unique(np.linspace(0,len(target)-1,min(12,len(target)),dtype=int)):
            value=time_flight(bc[i],target[i:i+1],pos[i,:10],vel[i,:10],v0_candidates=speeds,theta_candidates=angles,v_max=vmax,a_max=7.,**FLIGHT)
            actual=float(np.asarray(value)[0]);cached=float(t_ball[i])
            if np.isfinite(actual)!=np.isfinite(cached) or (np.isfinite(actual) and abs(actual-cached)>2e-5):raise ValueError(f'Flight cache/reference parity failed at {i}: {actual}, {cached}')
            checks.append(dict(row=int(i),t_ball=cached if np.isfinite(cached) else None,reference=actual if np.isfinite(actual) else None))
    return result,checks

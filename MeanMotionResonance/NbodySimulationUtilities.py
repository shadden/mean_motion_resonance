import numpy as np
import rebound as rb
import reboundx as rbx

def set_timestep(sim,dtfactor):
        ps=sim.particles[1:]
        tperi=np.min([p.P * (1-p.e)**1.5 / np.sqrt(1+p.e) for p in ps])
        dt = tperi * dtfactor
        sim.dt = dt
def set_min_distance(sim,rhillfactor):
        ps=sim.particles[1:]
        mstar = sim.particles[0].m
        rhill = np.min([ p.rhill for p in ps if p.m > 0])
        mindist = rhillfactor * rhill
        sim.exit_min_distance = mindist

def get_simarchive_integration_results(sa):
    if type(sa) == rb.simulationarchive.SimulationArchive:
        return _get_rebound_simarchive_integration_results(sa)
    elif type(sa) == rbx.simulationarchive.SimulationArchive:
        return _get_reboundx_simarchive_integration_results(sa)
    raise TypeError("{} is not a rebound or reboundx simulation archive!".format(sa))

def _get_rebound_simarchive_integration_results(sa):
    N = len(sa)
    sim0 = sa[0]
    Npl= sim0.N_real - 1
    shape = (Npl,N)
    sim_results = {
        'time':np.zeros(N),
        'P':np.zeros(shape),
        'e':np.zeros(shape),
        'l':np.zeros(shape),
        'inc':np.zeros(shape),
        'pomega':np.zeros(shape),
        'omega':np.zeros(shape),
        'Omega':np.zeros(shape),
        'a':np.zeros(shape),
        'Energy':np.zeros(N)
    }
    for i,sim in enumerate(sa):
        sim_results['time'][i] = sim.t
        orbits= sim.calculate_orbits(jacobi_masses=True)
        sim_results['Energy'][i] = sim.calculate_energy()
        for j,orbit in enumerate(orbits):
            sim_results['P'][j,i] = orbit.P
            sim_results['e'][j,i] = orbit.e
            sim_results['l'][j,i] = orbit.l
            sim_results['pomega'][j,i] = orbit.pomega
            sim_results['a'][j,i] = orbit.a
            sim_results['omega'][j,i] = orbit.omega
            sim_results['Omega'][j,i] = orbit.Omega
            sim_results['inc'][j,i] = orbit.inc
    return sim_results

def _get_reboundx_simarchive_integration_results(sa):
    N = len(sa)
    sim0,_ = sa[0]
    Npl= sim0.N_real - 1
    shape = (Npl,N)
    sim_results = {
        'time':np.zeros(N),
        'P':np.zeros(shape),
        'e':np.zeros(shape),
        'l':np.zeros(shape),
        'inc':np.zeros(shape),
        'pomega':np.zeros(shape),
        'omega':np.zeros(shape),
        'Omega':np.zeros(shape),
        'a':np.zeros(shape),
        'Energy':np.zeros(N)
    }
    for i,sim_extra in enumerate(sa):
        sim,extra = sim_extra
        sim_results['time'][i] = sim.t
        orbits= sim.calculate_orbits(jacobi_masses=True)
        sim_results['Energy'][i] = sim.calculate_energy()
        for j,orbit in enumerate(orbits):
            sim_results['P'][j,i] = orbit.P
            sim_results['e'][j,i] = orbit.e
            sim_results['l'][j,i] = orbit.l
            sim_results['pomega'][j,i] = orbit.pomega
            sim_results['a'][j,i] = orbit.a
            sim_results['omega'][j,i] = orbit.omega
            sim_results['Omega'][j,i] = orbit.Omega
            sim_results['inc'][j,i] = orbit.inc
    return sim_results

def get_canonical_heliocentric_orbits(sim):
    """
    Compute orbital elements in canconical 
    heliocentric coordinates.
    """
    star = sim.particles[0]
    orbits = []
    fictitious_star = rb.Particle(m=star.m)
    for planet in sim.particles[1:]:

        # Heliocentric position
        r = np.array(planet.xyz) - np.array(star.xyz)

        # Barycentric momentum
        rtilde = planet.m * np.array(planet.vxyz)

        # Mapping from (coordinate,momentum) pair to
        # orbital elments requires that the 'velocity'
        # i defined as the canonical momentum divided
        # by 'mu' appearing in the definition of the
        # Delauney variables( see, e.g., pg. 34 of
        # Morbidelli 2002).
        #
        # For Laskar & Robutel (1995)'s definition
        # of canonical action-angle pairs, this is
        # the reduced mass.
        #
        # For democratic heliocentric elements,
        # 'mu' is simply the planet mass.
        mu = planet.m * star.m / (planet.m + star.m)
        v_for_orbit = rtilde / mu

        fictitious_particle =  rb.Particle(
            m=planet.m,
            x = r[0],
            y = r[1],
            z = r[2],
            vx = v_for_orbit[0],
            vy = v_for_orbit[1],
            vz = v_for_orbit[2]
        )
        orbit = fictitious_particle.calculate_orbit(primary=fictitious_star,G = sim.G)
        orbits.append(orbit)
    return orbits

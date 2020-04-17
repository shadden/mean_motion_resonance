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

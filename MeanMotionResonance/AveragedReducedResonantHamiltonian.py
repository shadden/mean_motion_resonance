import rebound as rb
import reboundx
import numpy as np
import theano
import theano.tensor as T
from exoplanet.theano_ops.kepler import KeplerOp
from warnings import warn
from scipy.optimize import lsq_linear
DEBUG = False

def EulerAnglesTransform(x,y,z,Omega,I,omega):
    
    s1,c1 = T.sin(omega),T.cos(omega)
    x1 = c1 * x - s1 * y
    y1 = s1 * x + c1 * y
    z1 = z
    
    s2,c2 = T.sin(I),T.cos(I)
    x2 = x1
    y2 = c2 * y1 - s2 * z1
    z2 = s2 * y1 + c2 * z1

    s3,c3 = T.sin(Omega),T.cos(Omega)
    x3 = c3 * x2 - s3 * y2
    y3 = s3 * x2 + c3 * y2
    z3 = z2

    return x3,y3,z3

def _get_Omega_matrix(n):
    """
    Get the 2n x 2n skew-symmetric block matrix:
          [0 , I_n]
          [-I_n, 0 ]
    that appears in Hamilton's equations.

    Arguments
    ---------
    n : int
        Determines matrix dimension

    Returns
    -------
    numpy.array
    """
    return np.vstack(
        (
         np.concatenate([np.zeros((n,n)),np.eye(n)]).T,
         np.concatenate([-np.eye(n),np.zeros((n,n))]).T
        )
    )

def _get_compiled_theano_functions(N_QUAD_PTS):
    # Planet masses: m1,m2
    m1,m2 = T.dscalars(2)
    mstar = 1
    mu1  = m1 * mstar / (mstar  + m1) 
    mu2  = m2 * mstar / (mstar  + m2) 
    eta1 = mstar + m1
    eta2 = mstar + m2
    beta1 = mu1 * T.sqrt(eta1/mstar) / (mu1 + mu2)
    beta2 = mu2 * T.sqrt(eta2/mstar) / (mu1 + mu2)
    j,k = T.lscalars('jk')
    s = (j-k) / k

    # Angle variable for averaging over
    psi = T.dvector('psi')

    # Quadrature weights
    quad_weights = T.dvector('w')

    # Dynamical variables:
    dyvars = T.vector()
    s1, s2, phi, Omega, I1, I2, Phi, Rtilde, Psi  = [dyvars[i] for i in range(9)]

    Ndof = 4
    Nconst = 1
    
    a20 = T.constant(1.)
    a10 = ((j-k)/j)**(2/3) * (eta1 / eta2)**(1/3) * a20
    #Rtilde = -1 * (beta1 * T.sqrt(a10) + beta2 * T.sqrt(a20))
    #Omega = T.constant(0.)
    l1 = phi - 0.5 * k * psi
    l2 = phi + 0.5 * k * psi
    gamma1 = s1 - (1 + s) * l2 + s * l1
    gamma2 = s2 - (1 + s) * l2 + s * l1
    Gamma1 = I1
    Gamma2 = I2
    L1 = Phi / 2 - Psi / k - s * (I1 + I2)
    L2 = Phi / 2 + Psi / k + (s+1) * (I1 + I2)
    Cz = -1 * Rtilde

    R = L1+L2-Gamma1-Gamma2-Cz
    G1 = L1 - Gamma1
    G2 = L2 - Gamma2
    
    r2_by_r1 = (L2 - L1 - Gamma2 + Gamma1) / (L1 + L2 - Gamma1 - Gamma2 - R)
    rho1 = 0.5 * R * (1 + r2_by_r1)
    rho2 = 0.5 * R * (1 - r2_by_r1)


    a1 = (L1 / beta1 )**2 
    e1 = T.sqrt(1-(1-(Gamma1 / L1))**2)
    
    a2 = (L2 / beta2 )**2 
    e2 = T.sqrt(1-(1-(Gamma2 / L2))**2)
    
    cos_inc1 = 1 - rho1 / G1 
    cos_inc2 = 1 - rho2 / G2
    inc1 = T.arccos(cos_inc1)
    inc2 = T.arccos(cos_inc2)
    
    l1_r = l1 - Omega
    l2_r = l2 - Omega
    
    Omega1_r = T.constant(np.pi/2) - Omega
    Omega2_r = Omega1_r - T.constant(np.pi)
    
    pomega1 = -1 * gamma1
    pomega2 = -1 * gamma2
    
    pomega1_r = pomega1 - Omega
    pomega2_r = pomega2 - Omega

    omega1 = pomega1_r - Omega1_r
    omega2 = pomega2_r - Omega2_r

    Hkep = -0.5 * T.sqrt(eta1) * beta1 / a1 - 0.5 * T.sqrt(eta2) * beta2 / a2

    ko = KeplerOp()
    M1 = l1_r - pomega1_r
    M2 = l2_r - pomega2_r
    sinf1,cosf1 =  ko( M1, e1 + T.zeros_like(M1) )
    sinf2,cosf2 =  ko( M2, e2 + T.zeros_like(M2) )
    # 
    n1 = T.sqrt(eta1 / mstar ) * a1**(-3/2)
    n2 = T.sqrt(eta2 / mstar ) * a2**(-3/2)
    Hint_dir,Hint_ind,r1,r2,v1,v2 = calc_Hint_components_sinf_cosf(
            a1,a2,e1,e2,inc1,inc2,omega1,omega2,Omega1_r,Omega2_r,n1,n2,sinf1,cosf1,sinf2,cosf2
    )
    eps = m1*m2/(mu1 + mu2) / T.sqrt(mstar)
    Hpert = (Hint_dir + Hint_ind / mstar)
    Hpert_av = Hpert.dot(quad_weights)
    Htot = Hkep + eps * Hpert_av

    #####################################################
    # Set parameters for compiling functions with Theano
    #####################################################
    
    # Get numerical quadrature nodes and weights
    nodes,weights = np.polynomial.legendre.leggauss(N_QUAD_PTS)
    
    # Rescale for integration interval from [-1,1] to [-pi,pi]
    nodes = nodes * np.pi
    weights = weights * 0.5
    
    # 'givens' will fix some parameters of Theano functions compiled below
    givens = [(psi,nodes),(quad_weights,weights)]

    # 'ins' will set the inputs of Theano functions compiled below
    #   Note: 'extra_ins' will be passed as values of object attributes
    #   of the 'ResonanceEquations' class 'defined below
    extra_ins = [m1,m2,j,k]
    ins = [dyvars] + extra_ins
    orbels = [a1,e1,inc1,k*s1,a2,e2,inc2,k*s2,phi,Omega]
    orbels_dict = dict(zip(
            ['a1','e1','inc1','theta1','a2','e2','inc2','theta2','phi','Omega'],
            orbels
        )
    )
    #  Conservative flow
    gradHtot = T.grad(Htot,wrt=dyvars)
    hessHtot = theano.gradient.hessian(Htot,wrt=dyvars)
    Jtens = T.as_tensor(np.pad(_get_Omega_matrix(Ndof),(0,Nconst),'constant'))
    H_flow_vec = Jtens.dot(gradHtot)
    H_flow_jac = Jtens.dot(hessHtot)

    ##########################
    # Compile Theano functions
    ##########################
    orbels_fn = theano.function(
        inputs=ins,
        outputs=orbels_dict,
        givens=givens,
        on_unused_input='ignore'
    )
    Rtilde_fn = theano.function(
        inputs=ins,
        outputs=Rtilde,
        givens=givens,
        on_unused_input='ignore'
    )

    Htot_fn = theano.function(
        inputs=ins,
        outputs=Htot,
        givens=givens,
        on_unused_input='ignore'
    )

    Hpert_fn = theano.function(
        inputs=ins,
        outputs=Hpert_av,
        givens=givens,
        on_unused_input='ignore'
    )

    Hpert_components_fn = theano.function(
        inputs=ins,
        outputs=[Hint_dir.dot(quad_weights),Hint_ind.dot(quad_weights)],
        givens=givens,
        on_unused_input='ignore'
    )

    H_flow_vec_fn = theano.function(
        inputs=ins,
        outputs=H_flow_vec,
        givens=givens,
        on_unused_input='ignore'
    )
    
    H_flow_jac_fn = theano.function(
        inputs=ins,
        outputs=H_flow_jac,
        givens=givens,
        on_unused_input='ignore'
    )

    return dict({
        'orbital_elements':orbels_fn,
        'Rtilde':Rtilde_fn,
        'Hamiltonian':Htot_fn,
        'Hpert':Hpert_fn,
        'Hpert_components':Hpert_components_fn,
        'Hamiltonian_flow':H_flow_vec_fn,
        'Hamiltonian_flow_jacobian':H_flow_jac_fn
        })
def calc_Hint_components_sinf_cosf(a1,a2,e1,e2,inc1,inc2,omega1,omega2,Omega1,Omega2,n1,n2,sinf1,cosf1,sinf2,cosf2):
    """
    Compute the value of the disturbing function
    .. math::
        \frac{1}{|r-r'|} - ??? v.v'
    from a set of input orbital elements for coplanar planets.

    Arguments
    ---------
    a1 : float
        inner semi-major axis 
    a2 : float
        outer semi-major axis 
    e1 : float
        inner eccentricity
    e2 : float
        outer eccentricity
    I1 : float
        inner inclination
    I2 : float
        outer inclination
    omega1 : float
        inner argument of periapse
    omega2 : float
        outer argument of periapse
    dOmega : float
        difference in long. of nodes, Omega2-Omega1
    n1 : float
        inner mean motion
    n2 : float
        outer mean motion
    sinf1 : float
        sine of inner planet true anomaly
    cosf1 : float
        cosine of inner planet true anomaly
    sinf2 : float
        sine of outer planet true anomaly
    cosf2 : float
        cosine of outer planet true anomaly

    Returns
    -------
    (direct,indirect) : tuple
        Returns a tuple containing the direct and indirect parts
        of the interaction Hamiltonian
    """
    r1 = a1 * (1-e1*e1) /(1 + e1 * cosf1)
    _x1 = r1 * cosf1
    _y1 = r1 * sinf1
    _z1 = 0.
    x1,y1,z1 = EulerAnglesTransform(_x1,_y1,_z1,Omega1,inc1,omega1)

    vel1 = n1 * a1 / T.sqrt(1-e1*e1) 
    _u1 = -1 * vel1 * sinf1
    _v1 = vel1 * (e1 + cosf1)
    _w1 = 0.
    u1,v1,w1 = EulerAnglesTransform(_u1,_v1,_w1,Omega1,inc1,omega1)

    r2 = a2 * (1-e2*e2) /(1 + e2 * cosf2)
    _x2 = r2 * cosf2
    _y2 = r2 * sinf2
    _z2 = 0.
    x2,y2,z2 = EulerAnglesTransform(_x2,_y2,_z2,Omega2,inc2,omega2)
    vel2 = n2 * a2 / T.sqrt(2-e2*e2) 
    _u2 = -1 * vel2 * sinf2
    _v2 = vel2 * (e2 + cosf2)
    _w2 = 0.
    u2,v2,w2 = EulerAnglesTransform(_u2,_v2,_w2,Omega2,inc2,omega2)

    # direct term
    dx = x2 - x1
    dy = y2 - y1
    dz = z2 - z1 
    dr2 = dx*dx + dy*dy + dz*dz
    direct = -1 / T.sqrt(dr2)
    # indirect terms
    indirect = u1*u2 + v1*v2 + w1*w2
    return direct,indirect,[x1,y1,z1],[x2,y2,z2],[u1,v1,w1],[u2,v2,w2]

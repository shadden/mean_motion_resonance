import rebound as rb
import reboundx
import numpy as np
import theano
import theano.tensor as T
from exoplanet.theano_ops.kepler import KeplerOp
from warnings import warn
from scipy.optimize import lsq_linear
DEBUG = False
class ResonanceEquations():
    """
    A class for the model describing the dynamics of a pair of planar planets
    in/near a mean motion resonance.

    Includes the effects of dissipation.

    Attributes
    ----------
    j : int
        Together with k specifies j:j-k resonance
    
    k : int
        Order of resonance.
    
    alpha : float
        Semi-major axis ratio a_1/a_2

    eps : float
        Mass parameter m1*mu2 / (mu1+mu2)

    m1 : float
        Inner planet mass

    m2 : float
        Outer planet mass

    """
    pass

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

def simulation_to_dyvars(simulation,j,k):

def _get_compiled_theano_functions(N_QUAD_PTS):
    # resonance j and k
    j,k = T.lscalars('jk')
    s = (j-k) / k
    # Planet masses: m1,m2
    m1,m2 = T.dscalars(2)
    mstar = 1
    mu1  = m1 * mstar / (mstar  + m1) 
    mu2  = m2 * mstar / (mstar  + m2) 
    eta1 = mstar + m1
    eta2 = mstar + m2
    beta1 = mu1 * T.sqrt(eta1/mstar) / (mu1 + mu2)
    beta2 = mu2 * T.sqrt(eta2/mstar) / (mu1 + mu2)

    # Angle variable for averaging over
    Q = T.dvector('Q')
    
    # Dynamical variables:
    dyvars = T.vector()
    sigma1, sigma2, nu, I1, I2, N, amd = [dyvars[i] for i in range(7)]
    y1, y2, y_nu, x1, x2, y_nu, amd = [dyvars[i] for i in range(7)]

    I1 = (y1*y1 + x1*x1) / 2
    I2 = (y2*y2 + x2*x2) / 2
    N = (y_nu*y_nu + x_nu*x_nu) / 2
    sigma1 = T.arctan2(y1,x1)
    sigma2 = T.arctan2(y2,x2)
    nu = T.arctan2(y_nu,x_nu)
    
    # Quadrature weights
    quad_weights = T.dvector('w')
    
    # Set lambda2=0
    l2 = T.constant(0.)
    
    l1 = l2 - k * Q 
    pomega1 = (1+s) * l2 - s * l1 - sigma1
    pomega2 = (1+s) * l2 - s * l1 - sigma2
    
    Gamma1 = I1
    Gamma2 = I2
    
    # Resonant semi-major axis ratio
    a20 = T.constant(1.0)
    a10 = (eta1/eta2)**(1/3) * ((j-k)/j)**(2/3) * a20
    Lambda20 = beta2 * T.sqrt(a20)
    Lambda10 = beta1 * T.sqrt(a10)
    Ltot = Lambda10 + Lambda20
    # Choose z axis along angular momentum vector
    Lz = Ltot
    P0 =  0.5 * k * (Lambda20 - Lambda10)
    P = P0 - k * (s+1/2) * amd

    L1 = Ltot/2 - P / k - s * (I1 + I2)
    L2 = Ltot/2 + P / k + (1 + s) * (I1 + I2)

    Q1 = 0.5 *  k * Lz + N
    Q2 = 0.5 *  k * Lz - N

    
    a1 = (L1 / beta1 )**2 
    e1 = T.sqrt(1-(1-(Gamma1 / L1))**2)
    sinI1by2 = T.sqrt(0.5 * Q1 / (L1 - Gamma1)) 
    inc1 = 2 * T.arcsin(sinI1by2)
    
    a2 = (L2 / beta2 )**2 
    e2 = T.sqrt(1-(1-(Gamma2 / L2))**2)
    sinI2by2 = T.sqrt(0.5 * Q2 / (L2 - Gamma2)) 
    inc2 = 2 * T.arcsin(sinI2by2)
    
    Hkep = -0.5 * T.sqrt(eta1) * beta1 / a1 - 0.5 * T.sqrt(eta2) * beta2 / a2

    ko = KeplerOp()
    M1 = l1 - pomega1
    M2 = l2 - pomega2
    sinf1,cosf1 =  ko( M1, e1 + T.zeros_like(M1) )
    sinf2,cosf2 =  ko( M2, e2 + T.zeros_like(M2) )
    # nu = Omega2 - Omega1
    Omega2 = T.constant(0.)
    Omega1 = Omega2 - nu
    omega1 = pomega1 + Omega1
    omega2 = pomega2 + Omega2
    n1 = T.sqrt(eta1 / mstar ) * a1**(-3/2)
    n2 = T.sqrt(eta2 / mstar ) * a2**(-3/2)
    Hint_dir,Hint_ind = calc_Hint_sinf_cosf(
            a1,a2,e1,e2,inc1,inc2,omega1,omega2,Omega2-Omega1,n1,n2,sinf1,cosf1,sinf2,cosf2
    )

    eps = m1*m2/(mu1 + mu2) / T.sqrt(mstar * a20)
    Htot = Hkep + eps * (Hint_dir + Hint_ind / mstar)
    
    return dict({
        'a1':a1,
        'a2':a2,
        'e1':e1,
        'e2':e2,
        'inc1':inc1,
        'inc2':inc2,
        'Hkep':Hkep,
        'Hint_dir':Hint_dir,
        'Hint_ind':Hint_ind,
        'Htot':Htot
        })

def calc_Hint_components_sinf_cosf(a1,a2,e1,e2,inc1,inc2,omega1,omega2,dOmega,n1,n2,sinf1,cosf1,sinf2,cosf2):
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
    x1,y1,z1 = EulerAnglesTransform(_x1,_y1,_z1,0.,inc1,omega1)

    vel1 = n1 * a1 / T.sqrt(1-e1*e1) 
    _u1 = -1 * vel1 * sinf1
    _v1 = vel1 * (e1 + cosf1)
    _w1 = 0.
    u1,v1,w1 = EulerAnglesTransform(_u1,_v1,_w1,0.,inc1,omega1)

    r2 = a2 * (1-e2*e2) /(1 + e2 * cosf2)
    _x2 = r2 * cosf2
    _y2 = r2 * sinf2
    _z2 = 0.
    x2,y2,z2 = EulerAnglesTransform(_x2,_y2,_z2,dOmega,inc2,omega2)
    vel2 = n2 * a2 / T.sqrt(2-e2*e2) 
    _u2 = -1 * vel2 * sinf2
    _v2 = vel2 * (e2 + cosf2)
    _w2 = 0.
    u2,v2,w2 = EulerAnglesTransform(_u2,_v2,_w2,dOmega,inc2,omega2)

    # direct term
    dx = x2 - x1
    dy = y2 - y1
    dz = z2 - z1 
    dr2 = dx*dx + dy*dy + dz*dz
    direct = -1 / T.sqrt(dr2)
    # indirect terms
    indirect = u1*u2 + v1*v2 + w1*w2
    return direct,indirect

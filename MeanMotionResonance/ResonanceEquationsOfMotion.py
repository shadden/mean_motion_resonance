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

def canonical_heliocentric_orbital_elements_to_canonical_variables(m1,m2,a1,a2,e1,e2,inc1,inc2,lmbda1,lmbda2,omega1,omega2,Omega1,Omega2,j,k,mstar=1):
    mu1  = m1 * mstar / (mstar  + m1) 
    mu2  = m2 * mstar / (mstar  + m2) 
    eta1 = mstar + m1
    eta2 = mstar + m2
    beta1 = mu1 * np.sqrt(eta1/mstar) / (mu1 + mu2)
    beta2 = mu2 * np.sqrt(eta2/mstar) / (mu1 + mu2)

    Q = (lmbda2 - lmbda1) / k
    pomega1 = omega1 + Omega1
    pomega2 = omega2 + Omega2
    phi = j*lmbda2 - (j-k)*lmbda1  
    theta1 = phi - k * pomega1
    theta2 = phi - k * pomega2
    nu = 2*np.pi - k * (Omega2 + Omega1) / 2
    sigma1 = theta1 / k
    sigma2 = theta2 / k

    Ltot = mu1 * np.sqrt(eta1 * a1) * np.sqrt(1 - e1*e1) + mu2 * np.sqrt(eta2 * a2) * np.sqrt(1-e2*e2)
    factor  = mu2 + mu1 * (eta1 / eta2)**(2/3) * ((j-k)/j)**(1/3)
    a0 = (Ltot / factor)**2 / eta2
    Lmbda1 = beta1 * np.sqrt(a1 / a0)
    Lmbda2 = beta2 * np.sqrt(a2 / a0)

    I1 = Lmbda1 * (1 - np.sqrt(1 - e1*e1))
    I2 = Lmbda2 * (1 - np.sqrt(1 - e2*e2))

    G1 = Lmbda1 - I1
    G2 = Lmbda2 - I2
    G1x = G1 * np.sin(inc1) * np.sin(Omega1)
    G1y = -1 * G1 * np.sin(inc1) * np.cos(Omega1)
    G1z = G1 * np.cos(inc1)
    G2x = G2 * np.sin(inc2) * np.sin(Omega2)
    G2y = -1 * G2 * np.sin(inc2) * np.cos(Omega2)
    G2z = G2 * np.cos(inc2)
    G1vec = np.array([G1x,G1y,G1z]) 
    G2vec = np.array([G2x,G2y,G2z]) 
    Gtot_vec = G1vec + G2vec
    Gtot = np.sqrt(Gtot_vec @ Gtot_vec)
    cosi1 = G1vec @ Gtot_vec / G1 / Gtot
    cosi2 = G2vec @ Gtot_vec / G2 / Gtot
    Q1 =  G1 * (1-cosi1)
    Q2 =  G2 * (1-cosi2)
    
    N  = (Q1 + Q2)  / k

    rt2I1 = np.sqrt(2*I1)
    rt2I2 = np.sqrt(2*I2)
    rt2N  = np.sqrt(2*N)

    x1,y1 = rt2I1 * np.cos(sigma1), rt2I1 * np.sin(sigma1)
    x2,y2 = rt2I2 * np.cos(sigma2), rt2I2 * np.sin(sigma2)
    x_nu,y_nu = rt2N * np.cos(nu), rt2N * np.sin(nu)

    Lmbda10 = beta1 * ( eta1/eta2 )**(1/6) * ((j-k)/j)**(1/3)
    Lmbda20 = beta2  

    s = (j-k) / k
    P0 = k * (Lmbda20 - Lmbda10) / 2
    P  = k * (Lmbda2 - Lmbda1) / 2 - k * (s+1/2) * (I1 + I2)
    
    AMD = (P0 - P) / k /(s + 1/2) 
    
    return np.array([y1,y2,y_nu,x1,x2,x_nu,AMD]),a0,Q1,Q2,Ltot,P,P0

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
    y1, y2, y_nu, x1, x2, x_nu, amd = [dyvars[i] for i in range(7)]

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
    L = Lambda10 + Lambda20
    
    P0 =  0.5 * k * (Lambda20 - Lambda10)
    P = P0 - k * (s+1/2) * amd

    L1 = L/2 - P / k - s * (I1 + I2)
    L2 = L/2 + P / k + (1 + s) * (I1 + I2)

    # Choose z axis along direction of total angular momentum
    G1 = L1 - Gamma1
    G2 = L2 - Gamma2
    # Malige, Robutel, Laskar 2002 notation
    r1 = N / k
    r2_by_r1 = 0 # (G2 - G1) / (G1 + G2 - r1)
    Q1 = 0.5 * r1 * (1 + r2_by_r1)
    Q2 = 0.5 * r1 * (1 - r2_by_r1)

    a1 = (L1 / beta1 )**2 
    e1 = T.sqrt(1-(1-(Gamma1 / L1))**2)
    sinI1by2 = T.sqrt(0.5 * Q1 / G1 ) 
    inc1 = 2 * T.arcsin(sinI1by2)
    
    a2 = (L2 / beta2 )**2 
    e2 = T.sqrt(1-(1-(Gamma2 / L2))**2)
    sinI2by2 = T.sqrt(0.5 * Q2 / G2 ) 
    inc2 = 2 * T.arcsin(sinI2by2)
    
    Hkep = -0.5 * T.sqrt(eta1) * beta1 / a1 - 0.5 * T.sqrt(eta2) * beta2 / a2

    ko = KeplerOp()
    M1 = l1 - pomega1
    M2 = l2 - pomega2
    sinf1,cosf1 =  ko( M1, e1 + T.zeros_like(M1) )
    sinf2,cosf2 =  ko( M2, e2 + T.zeros_like(M2) )
    # 
    Omega2 = T.constant(-np.pi/2) - nu / k 
    Omega1 = Omega2 + T.constant(np.pi)
    omega1 = pomega1 - Omega1
    omega2 = pomega2 - Omega2
    n1 = T.sqrt(eta1 / mstar ) * a1**(-3/2)
    n2 = T.sqrt(eta2 / mstar ) * a2**(-3/2)
    Hint_dir,Hint_ind = calc_Hint_components_sinf_cosf(
            a1,a2,e1,e2,inc1,inc2,omega1,omega2,Omega1,Omega2,n1,n2,sinf1,cosf1,sinf2,cosf2
    )

    eps = m1*m2/(mu1 + mu2) / T.sqrt(mstar * a20)
    Hpert = (Hint_dir + Hint_ind / mstar).dot(quad_weights)
    Htot = Hkep + eps * Hpert

    #####################################################
    # Set parameters for compiling functions with Theano
    #####################################################
    
    # Get numerical quadrature nodes and weights
    nodes,weights = np.polynomial.legendre.leggauss(N_QUAD_PTS)
    
    # Rescale for integration interval from [-1,1] to [-pi,pi]
    nodes = nodes * np.pi
    weights = weights * 0.5
    
    # 'givens' will fix some parameters of Theano functions compiled below
    givens = [(Q,nodes),(quad_weights,weights)]
    
    # 'ins' will set the inputs of Theano functions compiled below
    #   Note: 'extra_ins' will be passed as values of object attributes
    #   of the 'ResonanceEquations' class 'defined below
    extra_ins = [m1,m2,j,k]
    ins = [dyvars] + extra_ins
    
    orbels = [a1,e1,inc1, sigma1 * k , Omega1, a2,e2,inc2,sigma2 * k, Omega2]

    #  Conservative flow
    gradHtot = T.grad(Htot,wrt=dyvars)
    hessHtot = theano.gradient.hessian(Htot,wrt=dyvars)
    Jtens = T.as_tensor(np.pad(_get_Omega_matrix(3),(0,1),'constant'))
    H_flow_vec = Jtens.dot(gradHtot)
    H_flow_jac = Jtens.dot(hessHtot)

    ##########################
    # Compile Theano functions
    ##########################
    orbels_fn = theano.function(
        inputs=ins,
        outputs=orbels,
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
        outputs=Hpert,
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

    R_fn = theano.function(
        inputs=ins,
        outputs=r2_by_r1 * r1,
        givens=givens,
        on_unused_input='ignore'
    )

    return dict({
        'orbital_elements':orbels_fn,
        'Hamiltonian':Htot_fn,
        'Hpert':Hpert_fn,
        'Hamiltonian_flow':H_flow_vec_fn,
        'Hamiltonian_flow_jacobian':H_flow_jac_fn,
        'R_constant':R_fn
        })

     #'Hkep':Hkep,
     #'Hint_dir':Hint_dir,
     #'Hint_ind':Hint_ind,
     #'Htot':Htot
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
    return direct,indirect

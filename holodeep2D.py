# 2D version of holodeep bead tracker, written for the DL formation 2019.
#
# Example usage:
#
# # create training set:
# X,Y = holodeep_3.make_training_test_set()
#
# # define and train the model:
# model = holodeep_3.define_model()
# model = holodeep_3.train_model(model, X,Y, epochs=20)
#
# # test on circular trajectory:
# holodeep_3.test_model(model, nimgs=200, ampl=1)


import numpy as np
import matplotlib.pyplot as plt


# physics parameters:
n_m  = 1.339        # refractive index medium
n_p  = 1.59         # index sphere (polystirene: 1.55-1.59):
lamb = 0.660        # illumination wavelength (microns)
umpp = 0.300        # camera micron (um) per pixel


def make_training_test_set():
    ''' return and save the training set, by making 'num_imgs' hologram images. '''    
    # parameters (modify them):
    num_imgs = 10000
    img_dim_px = [20,20]
    savename = './holodeep_trainset'
    img_scale = .7
    bead_rad_um = 0.5
    bead_zmin_um = -3.3
    bead_zmax_um = -3.3
    bead_noise = 0.05
    # init images:
    imgs = np.zeros((num_imgs, img_dim_px[0], img_dim_px[1]))
    # init data labels:
    pos  = np.zeros((num_imgs, 3))
    # make rand holograms images:
    for img_i in range(num_imgs):
        print('creating image: '+str(img_i)+' /'+str(num_imgs), end='\r')
        # make bead hologram:
        d_i = make_bead_holo(img_dim_px=img_dim_px, img_scale=img_scale, bead_rad_um=bead_rad_um, zmin_um=bead_zmin_um, zmax_um=bead_zmax_um, noise=bead_noise, plots=False)
        imgs[img_i,:,:] = d_i['holo_bead']
        pos[img_i,:]    = d_i['holo_bead_pos_px']
    # normalize the image data to mean 0 and std 1. 
    X = (imgs - np.mean(imgs)) / np.std(imgs)
    # normalize labels (x,y,z) so x,y are in 0,1:
    Y = pos/np.array((img_dim_px[0], img_dim_px[1], 1)) + np.array((0.5, 0.5, 0))
    Y = Y[:,:2]
    if savename:
        np.save(savename, {'X':X, 'Y':Y})
        print('saved: '+savename+'.npy')
    return X, Y



def define_model():
    ''' define and compile the model. 
    input_shape : include the channel as in [img_dim_px[0], img_dim_px[1], 1]
    label_shape : 3 or 2
    '''
    from keras.models import Sequential
    from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Activation, Dropout, BatchNormalization
    from keras import optimizers 
    from keras import regularizers
    label_shape = 2
    input_shape = (20,20,1)
    model = Sequential()
    # define model architecture:
    model.add(Conv2D(32, (3,3), activation='relu', padding='valid', input_shape=input_shape))
    model.add(Conv2D(64, (3,3), activation='relu', padding='valid', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(128, (3,3), activation='relu', padding='valid'))
    model.add(Conv2D(256, (3,3), activation='relu', padding='valid'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(256, activation='relu')) 
    model.add(Dense(128, activation='relu')) 
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(label_shape, activation=None))
    # compile model:
    model.compile(loss='mean_squared_error', optimizer='adamax', metrics=['accuracy'])
    return model



def train_model(model, X, Y, epochs=10):
    ''' returns trained model '''
    history = model.fit(x=X[:,:,:,np.newaxis], y=Y, epochs=epochs, batch_size=16, verbose=1, validation_split=0.2)
    plot_model(model)
    return model



def plot_model(model):
    ''' plot training history '''
    loss = model.history.history['loss']
    acc  = model.history.history['acc']
    val_loss = model.history.history['val_loss']
    val_acc  = model.history.history['val_acc']
    epochs = model.history.epoch
    fig = plt.figure('train_model()', clear=False)
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    ax1.semilogy(epochs, loss, label='loss')
    ax1.semilogy(epochs, val_loss, label='val_loss')
    ax1.legend()
    ax2.plot(epochs, acc, label='acc') 
    ax2.plot(epochs, val_acc, label='val_acc') 
    ax2.set_xlabel('epochs')
    ax2.legend()



def test_model(model, nimgs=1000, ampl=3):
    ''' a test set for the trained model'''
    X = np.zeros((nimgs, 20, 20, 1))
    Y = np.zeros((nimgs, 2))
    w = 0.1
    for i in range(nimgs):
        print(i, end='\r')
        x = ampl*np.sin(w*i)   
        y = ampl*np.cos(w*i)  
        z = -3.3/umpp #(- 6.5 - i/nimgs)/umpp
        d = make_bead_holo(xyzpos_px=[x,y,z])
        X[i,:,:,0] = d['holo_bead']
        Y[i,:] = d['holo_bead_pos_px'][:2]
    # normalize:
    X = (X - np.mean(X))/np.std(X)
    Y = Y/20 + 0.5
    # predict:
    pred = model.predict(X)
    # error (distance predictions - truth):
    err_rel = np.hypot(Y[:,0] - pred[:,0], Y[:,1] - pred[:,1])
    err_px = err_rel*20
    # plots:
    plt.figure('test_model()', clear=False)
    plt.subplot(211)
    plt.plot((Y[:,0]-0.5)*20, (Y[:,1]-0.5)*20, '.', label='truth')
    plt.plot((pred[:,0]-0.5)*20, (pred[:,1]-0.5)*20, 'o', alpha=0.2, label='prediction')
    plt.axis('scaled')
    plt.xlabel('px')
    plt.ylabel('px')
    plt.subplot(212)
    plt.hist(err_px, 20, alpha=0.3)
    plt.xlabel('error (px)')
    plt.ylabel('n.points')
    plt.title(f'mean error : {np.mean(err_px):.3f} px ({np.mean(err_px)*umpp:.3f} um)', fontsize=10)
    plt.tight_layout()



def make_bead_holo(img_dim_px=[20,20], img_scale=0.7, bead_rad_um=0.5, zmin_um=-6, zmax_um=-8, noise=0.05, xyzpos_px=[None,None,None], plots=False):
    ''' 
    return a dict with the hologram of a bead and its randomly chosen xyz position.
        img_dim_px          : w-h image dimensions in pixels.
        img_scale           : the center of the hologram falls within the area defined by img_dim_px * img_scale.
        bead_rad_um         : radius of the bead producing the hologram, in microns.
        zmin_um, zmax_um    : defines the range of the random z position of the bead.
        umpp                : micron per pixel.
        noise               : additive gaussian noise stdev 
        plots               : plot the hologram.
    '''
    # init xyz ranges:
    img_dim_px_scaled = np.array(img_dim_px) * img_scale
    zmin_px = zmin_um/umpp
    zmax_px = zmax_um/umpp
    if xyzpos_px == [None,None,None]:
        # def random hologram position:
        ra = np.random.random(size=3)
        holo_pos_px = (ra * np.array([img_dim_px_scaled[0]   ,  img_dim_px_scaled[1]  , zmax_px-zmin_px]) + \
                            np.array([-img_dim_px_scaled[0]/2, -img_dim_px_scaled[1]/2, zmin_px]))
    else:
        holo_pos_px = np.array(xyzpos_px)
    holo_pos_um = holo_pos_px*umpp
    # make hologram:
    holo = spheredhm(holo_pos_px, bead_rad_um, n_p, n_m, img_dim_px, lamb=lamb, mpp=umpp)
    holo = holo.T
    holo = holo + np.random.randn(img_dim_px[0], img_dim_px[1])*noise
    # make plot:
    if plots:
        plt.figure('rand_holo_position')
        plt.clf()
        plt.gray()
        plt.imshow(holo)
        plt.title(str(holo_pos_um))
        plt.colorbar()
    return {'holo_bead':holo, 'holo_bead_pos_px':holo_pos_px, 'holo_bead_pos_um':holo_pos_um}



#############################################################################
# Hologram computation adapted from https://github.com/markhannel/lorenzmie :
#############################################################################


def spheredhm(rp, a_p, n_p, n_m, dim, mpp = 0.135, lamb = .447, alpha = False, 
              precision = False,  lut = False):
    """
    Compute holographic microscopy image of a sphere immersed in a transparent 
    medium.

    Args:
        rp  : [x, y, z] 3 dimensional position of sphere relative to center
              of image.
        a_p  : radius of sphere [micrometers]
        n_p  : (complex) refractive index of sphere
        n_m  : (complex) refractive index of medium
        dim : [nx, ny] dimensions of image [pixels]

    NOTE: The imaginary parts of the complex refractive indexes
    should be positive for absorbing materials.  This follows the
    convention used in SPHERE_COEFFICIENTS.

    Keywords:
        precision: 
        alpha: fraction of incident light scattered by particle.
            Default: 1.
        lamb:  vacuum wavelength of light [micrometers]
        mpp: micrometers per pixel
        precision: relative precision with which fields are calculated.
    
    Returns:
        dhm: [nx, ny] holographic image                
    """
    
    nx, ny = dim
    x = np.tile(np.arange(nx, dtype = float), ny)
    y = np.repeat(np.arange(ny, dtype = float), nx)
    x -= float(nx)/2. + float(rp[0])
    y -= float(ny)/2. + float(rp[1])

    if lut:
        rho = np.sqrt(x**2 + y**2)
        x = np.arange(np.fix(rho).max()+1)
        y = 0. * x

    zp = float(rp[2])

    field = spherefield(x, y, zp, a_p, n_p, n_m = n_m, cartesian = True, mpp = mpp, 
                        lamb = lamb, precision = precision)
    #FP: here np.shape(field) = (3, nx*ny)
    
    if alpha: 
        field *= alpha
    
    k = 2.0*np.pi/(lamb/np.real(n_m)/mpp)
    
    # Compute the sum of the incident and scattered fields, then square.
    field *= np.exp(np.complex(0.,-k*zp))
    field[0,:] += 1.0
    image = np.sum(np.real(field*np.conj(field)), axis = 0)

    if lut: 
        image = np.interpolate(image, rho, cubic=-0.5)

    return image.reshape(int(ny), int(nx))







def spherefield(x, y, z, a_p, n_p, n_m = complex(1.3326, 1.5e-8),
                lamb = 0.447, mpp = 0.135, precision = False,
                cartesian = False):
    
    ab = sphere_coefficients(a_p, n_p, n_m, lamb)

    if precision:
        # retain first coefficient for bookkeeping
        fac = abs(ab[:, 1])
        w = np.where(fac > precision*max(fac))
        w = np.concatenate((np.array([0]),w[0]))
        ab =  ab[w,:]

    lamb_m = lamb/np.real(n_m)/mpp # medium wavelength [pixel]
    field = sphericalfield(x, y, z, ab, lamb_m, cartesian=cartesian,
                           str_factor = False)

    return field







def Nstop(x,m): 
    """
    Return the number of terms to keep in partial wave expansion
    according to Wiscombe (1980) and Yang (2003).
    """

    ### Wiscombe (1980)
    xl = x[-1]
    if xl < 8.:
        ns = np.floor(xl + 4. * xl**(1./3.) + 1.)
    elif xl < 4200.:
        ns = np.floor(xl + 4.05 * xl**(1./3.) + 2.)
    elif xl > 4199.:
        ns = np.floor(xl + 4. * xl**(1./3.) + 2.)

    ### Yang (2003) Eq. (30)
    ns = [ns]
    ns.extend(map(abs,x*m))
    ns.extend(map(abs,np.roll(x,-1)*m))
    return int(np.floor(max(ns))+15)



def sphere_coefficients(a_p, n_p, n_m, lamb, resolution=0):
    """
    Calculate the Mie scattering coefficients for a multilayered sphere
    illuminated by a coherent plane wave linearly polarized in the x direction.

    Args:
        a_p: [nlayers] radii of layered sphere [micrometers]
            NOTE: a_p and n_p are reordered automatically so that
            a_p is in ascending order.
        n_p: [nlayers] (complex) refractive indexes of sphere's layers
        n_m: (complex) refractive index of medium
        lamb: wavelength of light [micrometers]

    Keywrods:
        resolution: minimum magnitude of Lorenz-Mie coefficients to retain.
              Default: See references
    Returns:
        ab: the coefficients a,b
    """

    if type(a_p) != np.ndarray:
        a_p = [a_p,a_p]
        n_p = [n_p,n_p]
    a_p = np.array(a_p)
    n_p = np.array(n_p)

    nlayers = a_p.ndim

    if n_p.ndim != nlayers:
        print("Error Warning: a_p and n_p must have the same number of elements")

    # arrange shells in size order
    if nlayers > 1:
        order = a_p.argsort()
        a_p = a_p[order]
        n_p = n_p[order]

    x = map(abs,(2.0 * np.pi * n_m * a_p / lamb)) # size parameter [array]
    #FP:
    x = list(x)

    m = n_p/n_m                                # relative refractive index [array]
    nmax = Nstop(x, m)            # number of terms in partial-wave expansion
    ci = complex(0,1.)            # imaginary unit


    # arrays for storing results
    ab = np.zeros([nmax+1, 2],complex)  ##Note:  May be faster not to use zeros
    D1     = np.zeros(nmax+2,complex)
    D1_a   = np.zeros([nmax+2, nlayers],complex)
    D1_am1 = np.zeros([nmax+2, nlayers],complex)

    D3     = np.zeros(nmax+1,complex)
    D3_a   = np.zeros([nmax+1, nlayers],complex)
    D3_am1 = np.zeros([nmax+1, nlayers],complex)
    D3     = np.zeros(nmax+1,complex)
    D3_a   = np.zeros([nmax+1, nlayers],complex)
    D3_am1 = np.zeros([nmax+1, nlayers],complex)

    Psi         = np.zeros(nmax+1,complex)
    Zeta        = np.zeros(nmax+1,complex)
    PsiZeta     = np.zeros(nmax+1,complex)
    PsiZeta_a   = np.zeros([nmax+1, nlayers],complex)
    PsiZeta_am1 = np.zeros([nmax+1, nlayers],complex)

    Q  = np.zeros([nmax+1, nlayers],complex)
    Ha = np.zeros([nmax+1, nlayers],complex)
    Hb = np.zeros([nmax+1, nlayers],complex)

    # Calculate D1, D3 and PsiZeta for Z1 in the first layer
    z1 = x[0] * m[0]

    # D1_a[0, nmax + 1] = dcomplex(0) # Eq. (16a)
    for n in range(nmax+1, 0, -1):     # downward recurrence Eq. (16b)
        D1_a[n-1,0] = n/z1 - 1.0/(D1_a[n,0] + n/z1)

    PsiZeta_a[0, 0] = 0.5 * (1. - np.exp(2. * ci * z1)) # Eq. (18a)
    D3_a[0, 0] = ci                                  # Eq. (18a)
    for n in range(1, nmax+1):          #upward recurrence Eq. (18b)
        PsiZeta_a[n,0] = PsiZeta_a[n-1,0] * (n/z1 - D1_a[n-1,0]) * (n/z1 - D3_a[n-1,0])
        D3_a[n, 0] = D1_a[n, 0] + ci/PsiZeta_a[n, 0]

    # Ha and Hb in the core
    Ha[:, 0] = D1_a[0:-1, 0]     # Eq. (7a)
    Hb[:, 0] = D1_a[0:-1, 0]     # Eq. (8a)

    # Iterate from layer 2 to layer L
    for ii in range(1, nlayers):
        z1 = x[ii] * m[ii]
        z2 = x[ii-1] * m[ii]
        # Downward recurrence for D1, Eqs. (16a) and (16b)
        #   D1_a[ii, nmax+1]   = dcomplex(0)      # Eq. (16a)
        #   D1_am1[ii, nmax+1] = dcomplex(0)
        for n in range(nmax+1, 0, -1):# Eq. (16 b)
            D1_a[n-1, ii]   = n/z1 - 1./(D1_a[n, ii]   + n/z1)
            D1_am1[n-1, ii] = n/z2 - 1./(D1_am1[n, ii] + n/z2)

       # Upward recurrence for PsiZeta and D3, Eqs. (18a) and (18b)
        PsiZeta_a[0, ii]   = 0.5 * (1. - np.exp(2. * ci * z1)) # Eq. (18a)
        PsiZeta_am1[0, ii] = 0.5 * (1. - np.exp(2. * ci * z2))
        D3_a[0, ii]   = ci
        D3_am1[0, ii] = ci
        for n in range(1, nmax+1):    # Eq. (18b)
            PsiZeta_a[n, ii]   = PsiZeta_a[n-1, ii] * (n/z1 -  D1_a[n-1, ii]) * (n/z1 -  D3_a[n-1, ii])
            PsiZeta_am1[n, ii] = PsiZeta_am1[n-1, ii] * (n/z2 - D1_am1[n-1, ii]) * (n/z2 - D3_am1[n-1, ii])
            D3_a[n, ii]   = D1_a[n, ii]   + ci/PsiZeta_a[n, ii]
            D3_am1[n, ii] = D1_am1[n, ii] + ci/PsiZeta_am1[n, ii]
   # Upward recurrence for Q
        Q[0,ii] = (np.exp(-2. * ci * z2) - 1.) / (np.exp(-2. * ci * z1) - 1.)
        for n in range(1, nmax+1):
            Num = (z1 * D1_a[n, ii]   + n) * (n - z1 * D3_a[n-1, ii])
            Den = (z2 * D1_am1[n, ii] + n) * (n - z2 * D3_am1[n-1, ii])
            Q[n, ii] = (x[ii-1]/x[ii])**2 * Q[n-1, ii] * Num/Den

   # Upward recurrence for Ha and Hb, Eqs. (7b), (8b) and (12) - (15)
        for n in range(1, nmax+1):
            G1 = m[ii] * Ha[n, ii-1] - m[ii-1] * D1_am1[n, ii]
            G2 = m[ii] * Ha[n, ii-1] - m[ii-1] * D3_am1[n, ii]
            Temp = Q[n, ii] * G1
            Num = G2 * D1_a[n, ii] - Temp * D3_a[n, ii]
            Den = G2 - Temp
            Ha[n, ii] = Num/Den

            G1 = m[ii-1] * Hb[n, ii-1] - m[ii] * D1_am1[n, ii]
            G2 = m[ii-1] * Hb[n, ii-1] - m[ii] * D3_am1[n, ii]
            Temp = Q[n, ii] * G1
            Num = G2 * D1_a[n, ii] - Temp * D3_a[n, ii]
            Den = G2 - Temp
            Hb[n, ii] = Num/Den

       #ii (layers)

    z1 = complex(x[-1])
    # Downward recurrence for D1, Eqs. (16a) and (16b)
    # D1[nmax+1] = dcomplex(0)            # Eq. (16a)
    for n in range(nmax, 0, -1):         # Eq. (16b)
        D1[n-1] = n/z1 - (1./(D1[n] + n/z1))

    # Upward recurrence for Psi, Zeta, PsiZeta and D3, Eqs. (18a) and (18b)
    Psi[0]     =  np.sin(z1)                 # Eq. (18a)
    Zeta[0]    = -ci * np.exp(ci * z1)
    PsiZeta[0] =  0.5 * (1. - np.exp(2. * ci * z1))
    D3[0]      = ci
    for n in range(1, nmax+1):           # Eq. (18b)
        Psi[n]  = Psi[n-1]  * (n/z1 - D1[n-1])
        Zeta[n] = Zeta[n-1] * (n/z1 - D3[n-1])
        PsiZeta[n] = PsiZeta[n-1] * (n/z1 -D1[n-1]) * (n/z1 - D3[n-1])
        D3[n] = D1[n] + ci/PsiZeta[n]

    # Scattering coefficients, Eqs. (5) and (6)
    n = np.arange(nmax+1)
    ab[:, 0]  = (Ha[:, -1]/m[-1] + n/x[-1]) * Psi  - np.roll(Psi,  1) # Eq. (5)
    ab[:, 0] /= (Ha[:, -1]/m[-1] + n/x[-1]) * Zeta - np.roll(Zeta, 1)
    ab[:, 1]  = (Hb[:, -1]*m[-1] + n/x[-1]) * Psi  - np.roll(Psi,  1) # Eq. (6)
    ab[:, 1] /= (Hb[:, -1]*m[-1] + n/x[-1]) * Zeta - np.roll(Zeta, 1)
    ab[0, :]  = complex(0.,0.)
    if (resolution > 0):
        w = abs(ab).sum(axis=1)
        ab = ab[(w>resolution),:]

    return ab



def check_if_numpy(x, char_x):
    ''' checks if x is a numpy array '''
    if type(x) != np.ndarray:
        print(char_x + ' must be an numpy array')
        return False
    else:
        return True



    
def sphericalfield(x, y, z, ab, lamb, cartesian=False, str_factor=False, 
                   convention='Bohren'):
    """
    Calculate the complex electric field (or electric field strength factor) 
    due to a Lorenz-Mie scatterer a height z [pixels] above the grid (sx, sy).

    Args:
        x: [npts] array of pixel coordinates [pixels]
        y: [npts] array of pixel coordinates [pixels]
        z: If field is required in a single plane, then
            z is the plane's distance from the sphere's center
            [pixels]. Otherwise it is an [npts] array of pixel coordinates 
            [pixels].
        ab: [2, nc] array of a and b scattering coefficients, where
            nc is the number of terms required for convergence.
        lamb: wavelength of light in medium [pixels]

    Keywords:
        cartesian: if True, field is expressed as (x, y, z) else (r, theta, phi)
        str_factor: if True, returned field is the electric field strength 
            factor
    Returns:
        field: [3, npts] scattered electric field or field strength factor
    """

    # Check that inputs are numpy arrays
    for var, char_var in zip([x,y,ab], ['x', 'y', 'ab']):
        if not check_if_numpy(var, char_var):
            print('x, y and ab must be numpy arrays')
            return None

    z = np.array(z) # In case it is a float or integer.
    type_z = type(z)
    if type_z != np.ndarray and type_z != int and type_z != float:
        print('z must be a float, int or numpy array.')
        return None

    signZ = np.sign(z)

    # Check the inputs are the right size
    if x.shape != y.shape:
        print('x has shape {} while y has shape {}'.format(x.shape, y.shape))
        print('and yet their dimensions must match.')
        return None

    npts = len(x)
    nc = len(ab[:,0])-1     # number of terms required for convergence
    # FP: here nc = 34, npts = n pixels in image
    #print(nc)
    #print(npts)

    k = 2.0 * np.pi / lamb # wavenumber in medium [pixel^-1]

    # convert to spherical coordinates centered on the sphere.
    # (r, theta, phi) is the spherical coordinate of the pixel
    # at (x,y) in the imaging plane at distance z from the
    # center of the sphere.
    rho   = np.sqrt(x**2 + y**2)
    r     = np.sqrt(rho**2 + z**2)
    theta = np.arctan2(rho, z)
    phi   = np.arctan2(y, x)
    costheta = np.cos(theta)
    sintheta = np.sin(theta)
    cosphi   = np.cos(phi)
    sinphi   = np.sin(phi)

    kr = k*r # reduced radial coordinate

    # starting points for recursive function evaluation ...
    # ... Riccati-Bessel radial functions, page 478
    sinkr = np.sin(kr)
    coskr = np.cos(kr)

    '''
    Particles above the focal plane create diverging waves described by
    Eq. (4.13) for $h_n^{(1)}(kr)$. These have z > 0. Those below the focal
    plane appear to be converging from the perspective of the camera. They are
    descrinbed by Eq. (4.14) for $h_n^{(2)}(kr)$, and have z < 0. We can select
    the appropriate case by applying the correct sign of the imaginary part of
    the starting functions...
    '''
    if convention == 'Bohren':
        factor = 1
    else:
        factor = -1
    xi_nm2 = coskr + factor * signZ*1.j*sinkr # \xi_{-1}(kr)
    xi_nm1 = sinkr - factor * signZ*1.j*coskr # \xi_0(kr)
    #xi_nm2 = coskr + 1.j*sinkr
    #xi_nm1 = sinkr - 1.j*coskr
    # ... angular functions (4.47), page 95
    pi_nm1 = 0.0                    # \pi_0(\cos\theta)
    pi_n   = 1.0                    # \pi_1(\cos\theta)

    # storage for vector spherical harmonics: [r,theta,phi]
    Mo1n = np.zeros([3, npts],complex)
    Ne1n = np.zeros([3, npts],complex)

    # storage for scattered field
    Es = np.zeros([3, npts],complex)

    # Compute field by summing multipole contributions
    for n in range(1, nc+1):

        # upward recurrences ...
        # ... Legendre factor (4.47)
        # Method described by Wiscombe (1980)
        swisc = pi_n * costheta
        twisc = swisc - pi_nm1
        tau_n = pi_nm1 - n * twisc  # -\tau_n(\cos\theta)

        # ... Riccati-Bessel function, page 478
        xi_n = (2.0*n - 1.0) * xi_nm1 / kr - xi_nm2    # \xi_n(kr)

        # vector spherical harmonics (4.50)
        #Mo1n[0, :] = 0               # no radial component
        Mo1n[1, :] = pi_n * xi_n     # ... divided by cosphi/kr
        Mo1n[2, :] = tau_n * xi_n    # ... divided by sinphi/kr

        dn = (n * xi_n)/kr - xi_nm1
        
        # FP: this is slow:
        Ne1n[0, :] = n*(n + 1.0) * pi_n * xi_n # ... divided by cosphi sintheta/kr^2
        Ne1n[1, :] = tau_n * dn      # ... divided by cosphi/kr
        Ne1n[2, :] = pi_n  * dn      # ... divided by sinphi/kr

        # prefactor, page 93
        En = 1.j**n * (2.0*n + 1.0) / n / (n + 1.0)
        
        # FP: this is slow(est):
        # the scattered field in spherical coordinates (4.45)
        Es += En * (1.j * ab[n,0] * Ne1n - ab[n,1] * Mo1n)

        # upward recurrences ...
        # ... angular functions (4.47)
        # Method described by Wiscombe (1980)
        pi_nm1 = pi_n
        pi_n = swisc + (n + 1.0) * twisc / n

        # ... Riccati-Bessel function
        xi_nm2 = xi_nm1
        xi_nm1 = xi_n

    # geometric factors were divided out of the vector
    # spherical harmonics for accuracy and efficiency ...
    # ... put them back at the end.
    if str_factor:
        # Compute the electric field strength factor by removing r-dependence.
        radialFactor = np.exp(-1.0j*kr) / k
    else:
        radialFactor = 1 / kr
    Es[0, :] *= cosphi * sintheta * radialFactor / kr
    Es[1, :] *= cosphi * radialFactor
    Es[2, :] *= sinphi * radialFactor

    # By default, the scattered wave is returned in spherical
    # coordinates.  Project components onto Cartesian coordinates.
    # Assumes that the incident wave propagates along z and 
    # is linearly polarized along x
    if cartesian:
        Ec = np.zeros([3, npts], complex)
        Ec += Es

        Ec[0, :] =  Es[0, :] * sintheta * cosphi
        Ec[0, :] += Es[1, :] * costheta * cosphi
        Ec[0, :] -= Es[2, :] * sinphi

        Ec[1, :] =  Es[0, :] * sintheta * sinphi
        Ec[1, :] += Es[1, :] * costheta * sinphi
        Ec[1, :] += Es[2, :] * cosphi


        Ec[2, :] =  Es[0, :] * costheta - Es[1, :] * sintheta

        return Ec
    else:
        return Es


import gzip
import torch
import struct
import numpy as np

def coordinates_to_dmatrix(a_coords, b_coords, p=2):
    '''
    Creates distance matrix for numpy.array input.

    Parameters
    ----------
    a_coords : numpy.array
        numpy.array of shape (N,3) that contains coordinates information.

    b_coords : numpy.array
        numpy.array of shape (M,3) that contains coordinates information.

    Returns
    -------
    numpy.array of shape (N,M), the distance matrix of a_coords and b_coords.
    '''
    a, b = torch.from_numpy(a_coords), torch.from_numpy(b_coords)
    return np.array(torch.cdist(a,b,p=p))

def compute_geometric_diameters(coords):
    """
    Computes the min and max diameters of a 3D object using PCA:
    - Max diameter: along the longest axis (first principal component)
    - Min diameter: across the shortest axis (last principal component)
    
    Args:
        coords (np.ndarray): N x 3 atom coordinates.
    
    Returns:
        dict with min and max diameters in Angstroms.
    """
    if not len(coords):
        return {
        'min_diameter': 0,  # smallest axis
        'intermediate_diameter': 0,
        'max_diameter': 0   # longest axis
                    }
    # Center the coordinates
    coords_centered = coords - coords.mean(axis=0)

    # PCA: eigenvectors of the covariance matrix
    cov = np.cov(coords_centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)  # ascending order

    # Project onto principal components
    projected = coords_centered @ eigvecs

    # Diameters along each principal axis
    diameters = projected.max(axis=0) - projected.min(axis=0)

    return {
        'min_diameter': diameters[0],  # smallest axis
        'intermediate_diameter': diameters[1],
        'max_diameter': diameters[2]   # longest axis
    }

class MRC(object):
    """Load MRC file"""
    def __init__(self, filename):
        if filename.split('.')[-1] == 'gz':
            fin = gzip.open(filename, 'rb')
        else:
            fin = open(filename, 'rb')
        #with open(filename, 'rb') as fin:
        MRCdata = fin.read()
        self.nx = struct.unpack_from('<i', MRCdata, 0)[0]
        self.ny = struct.unpack_from('<i', MRCdata, 4)[0]
        self.nz = struct.unpack_from('<i', MRCdata, 8)[0]

        self.mode = struct.unpack_from('<i', MRCdata, 12)[0]
        #Starting point of sub image
        self.nxstart = struct.unpack_from('<i', MRCdata, 16)[0]
        self.nystart = struct.unpack_from('<i', MRCdata, 20)[0]
        self.nzstart = struct.unpack_from('<i', MRCdata, 24)[0]
        #Grid size in X, Y, and Z
        self.mx = struct.unpack_from('<i', MRCdata, 28)[0]
        self.my = struct.unpack_from('<i', MRCdata, 32)[0]
        self.mz = struct.unpack_from('<i', MRCdata, 36)[0]
        #Cell size; pixel spacing = xlen/mx, ylen/my, zlen/mz
        self.xlen = struct.unpack_from('<f', MRCdata, 40)[0]
        self.ylen = struct.unpack_from('<f', MRCdata, 44)[0]
        self.zlen = struct.unpack_from('<f', MRCdata, 48)[0]
        try:
            self.voxel = round(self.xlen/self.mx, 3)
        except:
            self.voxel = 0
        #cell angles
        self.alpha = struct.unpack_from('<f', MRCdata, 52)[0]
        self.beta = struct.unpack_from('<f', MRCdata, 56)[0]
        self.gamma = struct.unpack_from('<f', MRCdata, 60)[0]
        #MAP C R S 	axis corresp to cols, rows, sections respectively (1,2,3 for X,Y,Z)
        self.mapc = struct.unpack_from('<i', MRCdata, 64)[0]
        self.mapr = struct.unpack_from('<i', MRCdata, 68)[0]
        self.maps = struct.unpack_from('<i', MRCdata, 72)[0]
        #DMIN, DMAX, DMEAN, RMS
        self.dmin = struct.unpack_from('<f', MRCdata, 76)[0]
        self.dmax = struct.unpack_from('<f', MRCdata, 80)[0]
        self.dmean = struct.unpack_from('<f', MRCdata, 84)[0]
        self.rms = struct.unpack_from('<f', MRCdata, 216)[0]
        #Origin of image
        self.xorg = struct.unpack_from('<f', MRCdata, 196)[0]
        self.yorg = struct.unpack_from('<f', MRCdata, 200)[0]
        self.zorg = struct.unpack_from('<f', MRCdata, 204)[0]

        ind = self.nx*self.ny*self.nz*4
        self.data = np.frombuffer(MRCdata[-ind:], dtype=np.dtype(np.float32)).reshape((self.nx,self.ny,self.nz),order='F')
        fin.close()

class MRC_dimensions(object):
    """Load MRC file"""
    def __init__(self, filename):
        if filename.split('.')[-1] == 'gz':
            fin = gzip.open(filename, 'rb')
        else:
            fin = open(filename, 'rb')
        #with open(filename, 'rb') as fin:
        MRCdata = fin.read()
        self.nx = struct.unpack_from('<i', MRCdata, 0)[0]
        self.ny = struct.unpack_from('<i', MRCdata, 4)[0]
        self.nz = struct.unpack_from('<i', MRCdata, 8)[0]
        fin.close()

def write_mrc(rho, nxstart=0,nystart=0,nzstart=0, mapc=1,mapr=2,maps=3, xorg=0,yorg=0,zorg=0, alpha=90., beta=90., gamma=90., voxel_size=1.000, filename="map.mrc"):
    """Write an MRC formatted electron density map.
       See here: http://www2.mrc-lmb.cam.ac.uk/research/locally-developed-software/image-processing-software/#image
    """
    xs, ys, zs = rho.shape
    if type(voxel_size) is list or type(voxel_size) is tuple:
        a, b, c = xs*voxel_size[0], ys*voxel_size[1], zs*voxel_size[2]
    else:
        a, b, c = xs*voxel_size, ys*voxel_size, zs*voxel_size

    with open(filename, "wb") as fout:
        # NC, NR, NS, MODE = 2 (image : 32-bit reals)
        fout.write(struct.pack('<iiii', xs, ys, zs, 2))
        # NCSTART, NRSTART, NSSTART
        fout.write(struct.pack('<iii', nxstart, nystart, nzstart))
        # MX, MY, MZ
        fout.write(struct.pack('<iii', xs, ys, zs))
        # X length, Y, length, Z length
        fout.write(struct.pack('<fff', a, b, c))
        # Alpha, Beta, Gamma
        fout.write(struct.pack('<fff', alpha, beta, gamma))
        # MAPC, MAPR, MAPS
        fout.write(struct.pack('<iii', mapc, mapr, maps))
        # DMIN, DMAX, DMEAN
        fout.write(struct.pack('<fff', np.min(rho), np.max(rho), np.average(rho)))
        # ISPG, NSYMBT, mlLSKFLG
        fout.write(struct.pack('<iii', 1, 0, 0))
        # EXTRA
        fout.write(struct.pack('<'+'f'*12, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0))
        for i in range(0, 12):
            fout.write(struct.pack('<f', 0.0))

        # XORIGIN, YORIGIN, ZORIGIN
        fout.write(struct.pack('<fff', xorg, yorg, zorg )) #nxstart*(a/xs), nystart*(b/ys), nzstart*(c/zs) ))
        # MAP
        fout.write('MAP '.encode())
        # MACHST (little endian)
        fout.write(struct.pack('<BBBB', 0x44, 0x41, 0x00, 0x00))
        # RMS (std)
        fout.write(struct.pack('<f', np.std(rho)))
        # NLABL
        fout.write(struct.pack('<i', 0))
        # LABEL(20,10) 10 80-character text labels
        for i in range(0, 800):
            fout.write(struct.pack('<B', 0x00))
        # Write out data
        s = struct.pack('=%sf' % rho.size, *rho.flatten('F'))
        fout.write(s)

def normalize_image_array(im_arr: np.ndarray) -> np.ndarray:
    """
    Normalize each image in a batch to the range 0-1.
    
    Args:
        batch (np.ndarray): Input array of shape (batch, H, W)
        
    Returns:
        np.ndarray: Normalized array of the same shape with values in [0, 1]
    """
    min_vals = im_arr.min(axis=(1,2), keepdims=True)
    max_vals = im_arr.max(axis=(1,2), keepdims=True)
    
    # Avoid division by zero in case an image is constant
    denom = max_vals - min_vals
    denom = np.clip(denom, a_min=1e-4, a_max=None)
    
    normalized = (im_arr - min_vals) / denom
    return normalized

def normalize_single_image_array(im_arr: np.ndarray) -> np.ndarray:
    """
    Normalize each image in a batch to the range 0-1.
    
    Args:
        batch (np.ndarray): Input array of shape (batch, H, W)
        
    Returns:
        np.ndarray: Normalized array of the same shape with values in [0, 1]
    """
    min_vals = im_arr.min()
    max_vals = im_arr.max()
    
    # Avoid division by zero in case an image is constant
    denom = max_vals - min_vals
    denom = np.clip(denom, a_min=1e-4, a_max=None)
    
    normalized = (im_arr - min_vals) / denom
    return normalized

box_sizes = np.array([24, 32, 36, 40, 44, 48, 52, 56, 60, 64, 72, 84, 96, 100, 104, 112, 120, 128, 132, 140, 168, 180, 192, 196, 208, 216, 220, 224, 240, 256, 288, 300, 320, 352, 360, 384, 416, 440, 448, 480, 512, 540, 560, 576, 588, 600, 630, 640, 648, 672, 686, 700, 720, 750, 756, 768, 784, 800, 810, 840, 864, 882, 896, 900, 960, 972, 980, 1000, 1008, 1024, 1050, 1080, 1120, 1134, 1152, 1176, 1200, 1250, 1260, 1280, 1296, 1344, 1350, 1372, 1400, 1440, 1458, 1470, 1500, 1512, 1536, 1568, 1600, 1620, 1680, 1728, 1750, 1764, 1792, 1800, 1890, 1920, 1944, 1960, 2000, 2016, 2048, 2058, 2100, 2160, 2240, 2250, 2268, 2304, 2352, 2400, 2430, 2450, 2500, 2520, 2560, 2592, 2646, 2688, 2700, 2744, 2800, 2880, 2916, 2940, 3000, 3024, 3072, 3136, 3150, 3200, 3240, 3360, 3402, 3430, 3456, 3500, 3528, 3584, 3600, 3750, 3780, 3840, 3888, 3920, 4000, 4032, 4050, 4096])

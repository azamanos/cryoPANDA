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

class PDB(object):
    '''
    Class object to load PDB structure files of pdb or cif format.

    Parameters
    ----------
    filename : str
        Path to file.

    ignore_waters : bool
        To ignore or not the waters in the file, default False.

    ignore_other_HETATM : bool
        To ignore or not the HETATMs in the file, excluding waters, default False.

    multi_models : bool
        To include in the data or not multiple models of the structure in the file, default False.

    Attributes
    ----------
    natoms : int
        Number of protein atoms of all the chains in the structure.

    water_coords : numpy array
        Coordinates of water atoms, note that here we only keep oxygen atoms, shape (W,3).

    wb : numpy array
        B factor of water coordinates, shape (W).

    HETATM_coords : numpy array
        Coordinates of HETATM atoms, excluding water molecules, shape (H,3).

    HETATM_name : numpy array
        Names of HETATM atoms, excluding water molecules, shape (H).

    HETATM_num : numpy array
        Number of HETATM instance, excluding water molecules, shape (H).

    HETATM_atomnum : numpy array
        Atom number of HETATM atoms, excluding water molecules, shape (H).

    HETATM_atomtype : numpy array
        Atom type of HETATM atoms, excluding water molecules, shape (H).

    SSE : numpy array
        Empty numpy array ready to keep SSE info, shape (N).

    resolution : float
        Resolution of structure.

    SSEraw : numpy array
        Info of ranges for SSE info, shape (S,3)

    atomnum : numpy array
        Atom number of protein atoms, shape (N).

    atomname : numpy array
        Atom name of protein atoms, shape (N).

    atomalt : numpy array
        Alternative atoms for protein atoms, shape (N).

    resname : numpy array
        Residue name that protein atom belongs, shape (N).

    atomtype : numpy array
        Atom type of protein atoms, shape (N).

    resnum : numpy array
        Residue number of protein residues, shape (N).

    resalt : numpy array
        Alternative residues for protein residues, shape (N).

    chain : numpy array
        Chain that atom belongs, shape (N).

    coords : numpy array
        Atom coordinates of protein atoms, shape (N,3).

    occupancy : numpy array
        Occupancy of protein atoms, shape (N).

    b : numpy array
        B factor of protein atoms, shape (N).

    self.cella : float
        Unit cell parameters, length of a axis.

    self.cellb : float
        Unit cell parameters, length of b axis.

    self.cellc : float
        Unit cell parameters, length of c axis.

    self.cellalpha : float
        Unit cell parameters, angle of a axis.

    self.cellbeta : float
        Unit cell parameters, angle of b axis.

    self.cellgamma : float
        Unit cell parameters, angle of c axis.
    '''
    def __init__(self, filename, ignore_waters=False, ignore_other_HETATM=False, multi_models=False):
        #Define lists and variables
        self.natoms, self.water_coords, self.wb = 0, [], []
        self.HETATM_coords, self.HETATM_name, self.HETATM_chain, self.HETATM_num, self.HETATM_atomnum, self.HETATM_atomtype = [],[],[],[],[],[]
        self.SSE, self.resolution, self.SSEraw = [], None, []
        self.atomnum, self.atomname, self.atomalt, self.resname, self.atomtype, self.resnum, self.resalt, self.chain, self.coords = [],[],[],[],[],[],[],[],[]
        self.occupancy, self.b  = [],[]
        #Check if file is in cif format
        cif = False
        if filename.split('.')[-1] == 'cif' or filename.split('.')[-2] == 'cif':
            cif = True
        #If file is compressed uncompress it
        if filename.split('.')[-1] == 'gz':
            with gzip.open(filename, 'rt', encoding='utf-8') as file:
                f = file.readlines()
        #Else just open it
        else:
            with open(filename, 'r') as file:
                f = file.readlines()
        #If the format is pdb
        if not cif:
            #Start reading the PDB file
            for i, line in enumerate(f):
                sline = line.split()
                #If line starts with 'ATOM'
                if line[:4]=='ATOM':
                    #Keep protein's atoms info.
                    self.atomnum.append(int(float(sline[1])))
                    self.atomname.append(line[12:16].strip())
                    self.atomalt.append(line[16])
                    self.resname.append(line[17:21].strip())
                    atomtype = sline[-1]
                    if len(atomtype)-1:
                        try:
                            int(atomtype[1])
                            atomtype = atomtype[0]
                        except:
                            atomtype = atomtype[0].upper() + atomtype[1].lower()
                    self.atomtype.append(atomtype)
                    self.resnum.append(int(float(line[22:26])))
                    self.resalt.append(line[26])
                    self.chain.append(line[21])
                    self.coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
                    self.occupancy.append(float(line[56:60]))
                    self.b.append(float(line[60:66]))
                    #self.charge[atom] = line[78:80].strip('\n')
                    #self.nelectrons[atom] = electrons.get(self.atomtype[atom].upper(),6)
                    self.natoms += 1
                    continue
                #If line starts with 'HETATM'
                if line[:6] == 'HETATM':
                    #Keep waters
                    if not ignore_waters and line[13] == 'O' and ((line[17:20]=='HOH') or (line[17:20]=='TIP') or (line[17:20]=='WAT')):
                        self.water_coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
                        self.wb.append(float(line[60:66]))
                        continue
                    #Keep the rest hetatm
                    if not ignore_other_HETATM and sline[-1][0]!='H':
                        self.HETATM_coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
                        self.HETATM_name.append(line[17:21].strip())
                        self.HETATM_chain.append(line[21])
                        self.HETATM_num.append(int(float(line[22:26])))
                        self.HETATM_atomnum.append(int(float(line[6:12])))
                        atomtype = sline[-1]
                        if len(atomtype)-1:
                            try:
                                int(atomtype[1])
                                atomtype = atomtype[0]
                            except:
                                atomtype = atomtype[0].upper() + atomtype[1].lower()
                        self.HETATM_atomtype.append(atomtype)
                        continue
                #Here you can add more self objects from Header.
                #Resolution info in pdb format files.
                if line[:22] == 'REMARK   2 RESOLUTION.':
                    self.resolution = sline[3]
                    continue
                #Helix info in pdb format files.
                if line[:5] == 'HELIX':
                    self.SSEraw.append([range(int(line[20:25]),int(line[32:37])+1), line[19], 'H'])
                    continue
                #Beta sheet info in pdb format files.
                if line[:5] == 'SHEET':
                    self.SSEraw.append([range(int(line[22:26]),int(line[33:37])+1), line[21], 'S'])
                    continue
                #Crystal info in pdb format files.
                if line[:6] == 'CRYST1':
                    self.cella, self.cellb, self.cellc = float(sline[1]), float(sline[2]), float(sline[3])
                    self.cellalpha, self.cellbeta, self.cellgamma = float(sline[4]), float(sline[5]), float(sline[6])
                    continue
                #Return structure only of Model 1 if there are multiple Models
                if sline[0] == 'MODEL' and not cif and not multi_models and sline[1] != '1':
                    break
        #If the format is cif
        else:
            b_strands = False
            #Start reading the PDB file
            for i, line in enumerate(f):
                sline = line.split()
                #If line starts with 'ATOM'
                if line[:4]=='ATOM':
                    #First check if you have multiple models and you dont want them
                    if int(sline[-1])-1 and not multi_models:
                        break
                    self.atomnum.append(int(float(sline[1])))
                    self.atomname.append(sline[3])
                    self.atomalt.append(str(sline[4]))
                    self.resname.append(sline[5])
                    atomtype = sline[2]
                    if len(atomtype) == 2:
                        atomtype = atomtype[0].upper() + atomtype[1].lower()
                    self.atomtype.append(atomtype)
                    self.resnum.append(int(float(sline[-5])))
                    self.resalt.append(' ')
                    self.chain.append(sline[6])
                    self.coords.append([float(sline[10]), float(sline[11]), float(sline[12])])
                    self.occupancy.append(float(sline[13]))
                    self.b.append(float(sline[14]))
                    self.natoms += 1
                    continue
                #If line starts with 'HETATM'
                if line[:6] == 'HETATM':
                    if not ignore_waters and sline[2] == 'O' and ((sline[5]=='HOH') or (sline[5]=='TIP') or (sline[5]=='WAT')):
                        #Keep coordinates of water Oxygens.
                        self.water_coords.append([float(sline[10]), float(sline[11]), float(sline[12])])
                        self.wb.append(float(sline[14]))
                        continue
                    if not ignore_other_HETATM and sline[2]!='H':
                        self.HETATM_coords.append([float(sline[10]), float(sline[11]), float(sline[12])])
                        self.HETATM_name.append(sline[5])
                        self.HETATM_chain.append(sline[7])
                        try:
                            self.HETATM_num.append(int(float(sline[16])))
                        except:
                            self.HETATM_num.append(str(sline[16]))
                        self.HETATM_atomnum.append(int(float(sline[1])))
                        self.HETATM_atomtype.append(sline[2])
                        continue
                #Here you can add more self objects from Header.
                #Resolution info in cif format files.
                if line[:25] == '_reflns.d_resolution_high' or line[:33] == '_em_3d_reconstruction.resolution ':
                    self.resolution = sline[1]
                    continue
                #Helix info in cif format files.
                if line[:6] == 'HELX_P':
                    self.SSEraw.append([range(int(sline[-7]),int(sline[-4])+1), sline[-8], 'H'])
                    continue
                #Beta sheet info in cif format files.
                if line[:35] == '_struct_sheet_range.end_auth_seq_id':
                    b_strands = True
                    continue
                if b_strands:
                    if line[:1] == '#':
                        b_strands=False
                        continue
                    self.SSEraw.append([range(int(sline[-4]),int(sline[-1])+1), sline[-2], 'S'])
                    continue
                #Crystal info in cif format files.
                if line[:5] == '_cell':
                    if sline[0] == '_cell.length_a':
                        self.cella = float(sline[1])
                    if sline[0] == '_cell.length_b':
                        self.cellb = float(sline[1])
                    if sline[0] == '_cell.length_c':
                        self.cellc = float(sline[1])
                    if sline[0] == '_cell.angle_alpha':
                        self.cellalpha = float(sline[1])
                    if sline[0] == '_cell.angle_beta':
                        self.cellbeta = float(sline[1])
                    if sline[0] == '_cell.angle_gamma':
                        self.cellgamma = float(sline[1])
                    continue
        #Return structure when every atom and water atom have been searched.
        try:
            self.resolution = float(self.resolution)
        except:
            pass
        #Prepare SecondaryStructureElements info
        self.SSE = np.zeros((self.natoms), dtype=np.dtype((str,1)))
        self.SSEraw = np.array(self.SSEraw, dtype=object)
        #Turn every list to numpy array
        for attribute, value in vars(self).items():
            if type(value)==list:
                setattr(self, attribute, np.array(value))
        return

    def remove_waters(self):
        idx = np.where((self.resname=="HOH") | (self.resname=="TIP"))
        self.remove_atoms_from_object(idx)

    def remove_by_atomtype(self, atomtype):
        idx = np.where((self.atomtype==atomtype))
        self.remove_atoms_from_object(idx)

    def remove_by_atomname(self, atomname):
        idx = np.where((self.atomname==atomname))
        self.remove_atoms_from_object(idx)

    def remove_by_atomnum(self, atomnum):
        idx = np.where((self.atomnum==atomnum))
        self.remove_atoms_from_object(idx)

    def remove_by_resname(self, resname):
        idx = np.where((self.resname==resname))
        self.remove_atoms_from_object(idx)

    def remove_by_resnum(self, resnum):
        idx = np.where((self.resnum==resnum))
        self.remove_atoms_from_object(idx)

    def remove_by_chain(self, chain):
        idx = np.where((self.chain==chain))
        self.remove_atoms_from_object(idx)

    def remove_atoms_from_object(self, idx):
        mask = np.ones(self.natoms, dtype=bool)
        mask[idx] = False
        self.atomnum = self.atomnum[mask]
        self.atomname = self.atomname[mask]
        self.atomalt = self.atomalt[mask]
        self.resalt = self.resalt[mask]
        self.resname = self.resname[mask]
        self.resnum = self.resnum[mask]
        self.chain = self.chain[mask]
        self.coords = self.coords[mask]
        self.occupancy = self.occupancy[mask]
        self.b = self.b[mask]
        self.atomtype = self.atomtype[mask]
        self.SSE = self.SSE[mask]
        #self.charge = self.charge[mask]
        #self.nelectrons = self.nelectrons[mask]
        self.natoms = len(self.atomnum)

    def rearrange_resalt(self):
        #Keep indexes where you have added residues
        ind = np.where(self.resalt!=' ')[0]
        if not len(ind):
            return
        #Find chains of these added residues
        resalt_ch = self.chain[ind]
        #Find unique chains
        diff_chains = np.unique(resalt_ch)
        #For each unique chain id
        for ch in diff_chains:
            #Keep indexes of added residues only for your chain
            ind_resalt_ch = ind[np.where(resalt_ch==ch)]
            #Keep last index of your chain's residues
            last_ch_ind = np.where(self.chain==ch)[0][-1]+1
            #For each atom in chain in added residue
            for atom in ind_resalt_ch:
                #Find where added residue starts
                if self.resalt[atom-1]!=self.resalt[atom] and self.resnum[atom-1] == self.resnum[atom]:
                    #Add to all consquent residue +1 number
                    self.resnum[atom:last_ch_ind] += 1
                #If added residues are before residue number, change the numbering of the original residue +1
                if atom+1+1>self.natoms:
                    continue
                if self.resalt[atom+1] == ' ' and self.resnum[atom+1] == self.resnum[atom]:
                    self.resnum[atom+1:last_ch_ind] += 1
        #Remove alternative residues indexes
        self.resalt[ind] = ' '

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
        self.voxel = round(self.xlen/self.mx, 3)
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
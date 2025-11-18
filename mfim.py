import scipy.sparse as sparse
import numpy as np

class MFIM:
    '''
    Ising model with transverse and longitudinal fields
    '''

    def __init__(self,**kwargs):
        
        self.L = kwargs.pop('L',4)          # spin chain length
        self.J = kwargs.pop('J',10)         # coupling strength
        self.hz = kwargs.pop('hz',1)        # longitudinal field strength
        self.hx = kwargs.pop('hx',1)        # transverse field strength
        
        self.H = kwargs.pop('H',0)          # Hamiltonian
        self.sx_list = kwargs.pop('sx_list',0)  # Pauli X_i for each site
        self.sy_list = kwargs.pop('sy_list',0)  # Pauli Y_i for each site
        self.sz_list = kwargs.pop('sz_list',0)  # Pauli Z_i for each site
        self.I = kwargs.pop('I',0)          # Identity in given Hilbert space


    @classmethod
    def init(cls, **kwargs):
        '''
        Initialize mix-field Ising model (MFIM) Hamiltonian
        '''

        mfim = cls(**kwargs)

        mfim.get_H()

        return mfim

    def get_H(self):

        self.I, self.sx_list, self.sy_list, self.sz_list = self.gen_s0sxsysz()
        self.H = self.hz * sum(self.sz_list) + self.hx * sum(self.sx_list)

        if np.isscalar(self.J):
            for i in range(self.L-1):

                self.H += -self.J * self.sz_list[i] * self.sz_list[(i+1)]
        
        else:
            J12, J13, J14, J23, J24, J34 = self.J
            self.H += (   J12*self.sz_list[0]*self.sz_list[1]
                        + J13*self.sz_list[0]*self.sz_list[2]
                        + J14*self.sz_list[0]*self.sz_list[3]
                        + J23*self.sz_list[1]*self.sz_list[2]
                        + J24*self.sz_list[1]*self.sz_list[3]
                        + J34*self.sz_list[2]*self.sz_list[3])
            
        return 1
    
    def gen_s0sxsysz(self):
        sx = sparse.csr_matrix([[0., 1.],[1., 0.]])
        sy = sparse.csr_matrix([[0.,-1j],[1j,0.]])
        sz = sparse.csr_matrix([[1., 0],[0, -1.]])
        sx_list = []
        sy_list = []
        sz_list = []
        I = sparse.eye(2**self.L, format='csr', dtype='complex')
        for i_site in range(self.L):
            if i_site==0:
                X=sx
                Y=sy
                Z=sz
            else:
                X= sparse.csr_matrix(np.eye(2))
                Y= sparse.csr_matrix(np.eye(2))
                Z= sparse.csr_matrix(np.eye(2))

            for j_site in range(1,self.L):
                if j_site==i_site:
                    X=sparse.kron(X,sx, 'csr')
                    Y=sparse.kron(Y,sy, 'csr')
                    Z=sparse.kron(Z,sz, 'csr')
                else:
                    X=sparse.kron(X,np.eye(2),'csr')
                    Y=sparse.kron(Y,np.eye(2),'csr')
                    Z=sparse.kron(Z,np.eye(2),'csr')
            sx_list.append(X)
            sy_list.append(Y)
            sz_list.append(Z)

        return I, sx_list,sy_list,sz_list

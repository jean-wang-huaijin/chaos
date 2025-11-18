from scipy import sparse
from mfim import MFIM
import scipy.sparse.linalg as spalin
from scipy.sparse import csr_matrix, kron, eye
from functools import reduce
import numpy as np
import matplotlib.pyplot as plt

class OTOC:
    '''
    Extract specified OTOC C^{\mu\nu}_{ij}(t) between time interval [0,T]
    '''

    def __init__(self, **kwargs):
        
        self.mfim = kwargs.pop('mfim',0)                        # mix field Ising model
        self.init_state = kwargs.pop('init_state', "mixed")     # initial state: "mixed", "pure", "thermal"
        self.state_param = kwargs.pop('state_param',0)          # ε for mixed, bits for pure, T for thermal
        self.rho = kwargs.pop('rho',0)

        self.mu = kwargs.pop('mu','X')              # mu (first Pauli): 'X', 'Y', 'Z'
        self.nu = kwargs.pop('nu','X')              # nu (second Pauli): 'X', 'Y', 'Z'
        self.i = kwargs.pop('i',0)                  # i (first site): 0 to L-1
        self.j = kwargs.pop('j',0)                  # j (second site): 0 to L-1

        self.T = kwargs.pop('T',0)                  # T (end time)
        self.tstep = kwargs.pop('tstep',0)          # tstep
        self.Dt = kwargs.pop('Dt',0)                # time interval to extract chaos measure

        self.dynamics = kwargs.pop('dynamics','unitary')    # 'unitary' or 'dephasing' or 'amplitude_damping'
        self.location = kwargs.pop('location','boundary')   # location of dissipation: 'boundary' or 'bulk'
        self.kappa = kwargs.pop('kappa',0)                  # kappa for dephasing
        self.direction = kwargs.pop('direction','Z')        # direction for dephasing: 'X', 'Z'
        self.gamma = kwargs.pop('gamma',0)                  # gamma for amplitude damping
        self.n_th = kwargs.pop('n_th',0)                    # n_th for amplitude damping

        self.tlist = 0
        self.otoc_list = 0                          # specified OTOC between time interval [0,T]
        self.normalized_otoc_list = 0
        self.std = 0

    @classmethod
    def init(cls, **kwargs):

        L = kwargs.pop('L',4)
        J = kwargs.pop('J',0)
        hz = kwargs.pop('hz',1)
        hx = kwargs.pop('hx',1)

        mfim = MFIM.init(L=L, J=J, hz=hz, hx=hx)

        return cls(mfim=mfim, **kwargs)
    

    def analysis(self):
        '''
        Compute OTOC list, get standard deviation, extract chaos measure
        '''
        self.get_otoc_list()
        self.normalized_otoc_list, self.std = self.get_std()


    def analysis_normalized(self):
        '''
        Compute OTOC list, get standard deviation, extract chaos measure
        with normalization against identity operator
        '''
        original_mu = self.mu
        self.get_otoc_list()
        otoc_list_A = np.copy(self.otoc_list) 

        self.mu = 'I'
        self.get_otoc_list() 
        otoc_list_I = np.copy(self.otoc_list)
        self.mu = original_mu

        F_A = 1.0 - otoc_list_A
        F_I = 1.0 - otoc_list_I

        F_corr = np.divide(F_A, F_I, out=np.ones_like(F_A), where=F_I!=0)
        
        self.otoc_list = 1.0 - F_corr


        self.normalized_otoc_list, self.std = self.get_std()


    def get_otoc_list(self):

        self.tlist = np.arange(0 , self.T + self.tstep, self.tstep)
        self.otoc_list = np.zeros(len(self.tlist))

        for i, t in enumerate(self.tlist):
            self.otoc_list[i] = self.otoc_t(t)

        
    def get_std(self):
        ts = self.get_first_peak(self.otoc_list)
        time_window = self.otoc_list[ts+1:ts+int(self.Dt/(self.tlist[1]-self.tlist[0]))]
        otoc_norm = self.otoc_list/np.mean(time_window)
        std = np.sqrt(np.mean((time_window/np.mean(time_window))**2)-1)
        
        return otoc_norm, std
    
    
    def otoc_t(self, t, **kwargs):

        op_dict = {
            'X': self.mfim.sx_list,
            'Y': self.mfim.sy_list,
            'Z': self.mfim.sz_list,
            'I': self.mfim.I
        }

        if self.mu == 'I':
            A = op_dict['I']
        else:
            A = op_dict[self.mu][self.i]

        B = op_dict[self.nu][self.j]
        
        if self.init_state == "mixed":
            rho_pure = self.gen_pure_state([1],[[0,0,0,1]])
            self.rho = rho_pure*self.state_param + self.mfim.I/(2**self.mfim.L)*(1-self.state_param)
        elif self.init_state == "pure":
            self.rho = self.gen_pure_state(self.state_param[0],self.state_param[1])
        elif self.init_state == "thermal":
            self.rho = self.gen_thermal_state(self.state_param, self.mfim.H)
        else:
            self.rho = self.mfim.I/(2**self.mfim.L)

        if self.dynamics == 'unitary':
            U = spalin.expm(-1j * self.mfim.H * t) # Compute U(t) = exp(-i * H * t)
            U_dag = U.getH()  # Compute U^{\dagger}
            S_i_t = U_dag @ A @ U  # Compute the time-evolved operator S_i(t)
            product = self.rho @ S_i_t @ B @ S_i_t @ B  # Compute the product S_i(t) S_j S_i(t) S_j
            otoc = 1-np.real(product.diagonal().sum())   # Compute trace

        elif self.dynamics in ('dephasing', 'amplitude_damping'):
            d = self.mfim.H.shape[0]
            B_dag = B.getH()
            A_dag = A.getH()

            # V_b_dag(t) * B_dag
            # V_b_dag uses -H_s and Heisenberg Liouvillian
            v_B_dag = self.vec(B_dag)
            L_b_dag = self.build_heisenberg_liouvillian(-self.mfim.H) 
            v_B_back_t = spalin.expm_multiply(L_b_dag * t, v_B_dag)
            B_back_t = self.unvec(v_B_back_t, d)

            # V_f(t) * (B * rho)
            # V_f uses +H_s and Schrödinger Liouvillian
            rho_to_evolve = B @ self.rho
            v_rho_to_evolve = self.vec(rho_to_evolve)
            L_f = self.build_schrodinger_liouvillian(self.mfim.H) 
            v_rho_mid_t = spalin.expm_multiply(L_f * t, v_rho_to_evolve)
            rho_mid_t = self.unvec(v_rho_mid_t, d)

            product = B_back_t @ A @ rho_mid_t @ A_dag
            
            otoc_F = np.real(product.diagonal().sum()) # This is F(t, A, B)
            otoc = 1.0 - otoc_F                        # Return C(t) = 1 - F(t)

        return otoc

    def build_heisenberg_liouvillian(self, H_system):
        H = H_system.tocsr().astype(complex)
        d = H.shape[0]
        I = eye(d, format='csr', dtype=complex)

        # Hamiltonian part: i(I ⊗ H - H^T ⊗ I)  acting on vec(A)
        L_H = 1j * (kron(I, H, 'csr') - kron(H.T, I, 'csr'))

        for Lop in self.jump_operators():
            Lop = Lop.tocsr().astype(complex)
            Ldag = Lop.getH()
            LdagL = (Ldag @ Lop).tocsr()

            # vec(L^\dagger A L) = (L^T ⊗ L^\dagger) vec(A)
            term1 = kron(Lop.T, Ldag, 'csr')
            # -1/2 {L^\dagger L, A}  ->  -1/2 [ I ⊗ (L^\dagger L) + (L^\dagger L)^T ⊗ I ]
            term2 = -0.5 * kron(I, LdagL, 'csr')
            term3 = -0.5 * kron(LdagL.T, I, 'csr')

            L_H = (L_H + term1 + term2 + term3).tocsr()

        return L_H 

    def build_schrodinger_liouvillian(self, H_system):
        H = H_system.tocsr().astype(complex)
        d = H.shape[0]
        I = eye(d, format='csr', dtype=complex)

        L_S = -1j * (kron(I, H, 'csr') - kron(H.T, I, 'csr'))

        for Lop in self.jump_operators():
            Lop = Lop.tocsr().astype(complex)
            Ldag = Lop.getH()
            LdagL = (Ldag @ Lop).tocsr()

            term1 = kron(Lop.conj(), Lop, 'csr')
            term2 = -0.5 * kron(I, LdagL, 'csr')
            term3 = -0.5 * kron(LdagL.T, I, 'csr')

            L_S = (L_S + term1 + term2 + term3).tocsr()

        return L_S

    def jump_operators(self):
        if self.dynamics == "dephasing":
            if self.direction == 'Z':
                if self.location == 'boundary':
                    # acting sigma_z on site 0 and L-1
                    ops = [self.mfim.sz_list[0], self.mfim.sz_list[self.mfim.L-1]]
                else:
                    # acting sigma_z on all sites
                    ops = self.mfim.sz_list
            elif self.direction == 'X':
                if self.location == 'boundary':
                    # acting sigma_x on site 0 and L-1
                    ops = [self.mfim.sx_list[0], self.mfim.sx_list[self.mfim.L-1]]
                else:
                    # acting sigma_x on all sites
                    ops = self.mfim.sx_list

            return [(np.sqrt(self.kappa) * op).tocsr().astype(complex) for op in ops]
        
        elif self.dynamics == "amplitude_damping":
            if self.location == "boundary":
                sites = [0, self.mfim.L - 1]
            else:
                sites = range(self.mfim.L)

            ops = []
            for i in sites:
                sx_i = self.mfim.sx_list[i]
                sy_i = self.mfim.sy_list[i]
                sp_i = 0.5 * (sx_i + 1j * sy_i)   # sigma_+
                sm_i = 0.5 * (sx_i - 1j * sy_i)   # sigma_-
                if self.n_th > 0:
                    ops.append(np.sqrt(self.gamma * self.n_th) * sp_i)        # absorption
                ops.append(np.sqrt(self.gamma * (self.n_th + 1.0)) * sm_i)    # emission

            return [op.tocsr().astype(complex) for op in ops]
    
    def plot_otoc(self):

        plt.figure(figsize=(8,4))
        plt.plot(self.tlist,self.normalized_otoc_list, label = f'L = {self.mfim.L}, hz = {self.mfim.hz}')
        plt.axhline(y=1, color='black', linestyle='--')
        plt.xlabel('$t$')
        plt.ylabel(rf"$C^{{{self.mu}{self.nu}}}_{{{self.i}{self.j}}}$")
        plt.legend()
        plt.show()
    

    @staticmethod
    def get_first_peak(lst):

        for i in range(1, len(lst) - 1):
            if lst[i] > lst[i - 1] and lst[i] > lst[i + 1]:
                return i 
            else:
                return 0
            
    @staticmethod
    def gen_thermal_state(temp, H, k_B=1):

        beta = 1/(k_B * temp)
        exp = spalin.expm(-beta * H)
        Z = exp.diagonal().sum()
        rho = exp/Z

        return rho
    
    @staticmethod
    def gen_pure_state(coeffs, bits):
        '''
        Generate density matrix ρ=∣ψ⟩⟨ψ∣ of a pure state given ψ
        '''
        
        ket0 = csr_matrix([[1],[0]])
        ket1 = csr_matrix([[0],[1]])

        norm_coeffs = coeffs/np.sqrt(np.sum(np.abs(coeffs)**2))

        psi = 0

        for i,c in enumerate(norm_coeffs):
            psi += c*reduce(kron, [(ket0 if b==0 else ket1) for b in bits[i]])
            
        rho = psi @ psi.getH()
        
        return rho

    @staticmethod
    def vec(A):
        # convert a matrix to a vector ((a,b),(c,d)) -> (a,c,b,d)
        return np.asarray(A.toarray(), dtype=complex).reshape((-1, 1), order='F')

    @staticmethod
    def unvec(v, d):
        # convert a vector to a matrix (a,c,b,d) -> ((a,b),(c,d))
        return sparse.csr_matrix(v.reshape((d, d), order='F'))


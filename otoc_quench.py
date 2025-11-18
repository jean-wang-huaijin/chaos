import scipy
from mfim import MFIM
import scipy.sparse.linalg as spalin
from scipy.sparse import csr_matrix, kron
from functools import reduce
import ast
import numpy as np
import matplotlib.pyplot as plt

class OTOC:
    '''
    Extract specified OTOC C^{\mu\nu}_{ij}(t) between time interval [0,T]
    '''

    def __init__(self, **kwargs):
        
        self.mfim = kwargs.pop('mfim',0)            # mix field Ising model
        self.init_state = kwargs.pop('init_state', "mixed")    # initial state: "mixed", "pure", "thermal"
        self.state_param = kwargs.pop('state_param',0)          # ε for mixed, bits for pure, T for thermal
        self.rho = kwargs.pop('rho',0)

        self.mu = kwargs.pop('mu','X')              # mu (first Pauli): 'X', 'Y', 'Z'
        self.nu = kwargs.pop('nu','X')              # nu (second Pauli): 'X', 'Y', 'Z'
        self.i = kwargs.pop('i',0)                  # i (first site): 0 to L-1
        self.j = kwargs.pop('j',0)                  # j (second site): 0 to L-1

        self.T = kwargs.pop('T',0)                  # T (end time)
        self.tstep = kwargs.pop('tstep',0)          # tstep
        self.Dt = kwargs.pop('Dt',0)                # time interval to extract chaos measure
        self.quench_params = kwargs.pop('quench_params', None) # quench parameter

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

    def get_initial_rho(self):
        if self.init_state == "mixed":
            rho_pure = self.gen_pure_state([1],[[0,0,0,1]])
            rho = rho_pure*self.state_param + self.mfim.I/(2**self.mfim.L)*(1-self.state_param)
        elif self.init_state == "pure":
            rho = self.gen_pure_state(self.state_param[0],self.state_param[1])
        elif self.init_state == "thermal":
            rho = self.gen_thermal_state(self.state_param, self.mfim.H)
        else:
            rho = self.mfim.I/(2**self.mfim.L)
        return rho


    def get_otoc_list(self):
        self.tlist = np.arange(0 , self.T + self.tstep, self.tstep)
        self.otoc_list = np.zeros(len(self.tlist))

        # --- 1. Get initial operators and state (from otoc_t) ---
        op_dict = {
            'X': self.mfim.sx_list,
            'Y': self.mfim.sy_list,
            'Z': self.mfim.sz_list
        }
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

        # --- 2. Initialize evolving objects ---
        current_H = self.mfim.H
        current_hz = self.mfim.hz
        U_step = None
        current_A_t = A # A(t=0) = A
        quench_segment = 0
        
        # --- 3. Start time-stepping loop ---
        for i, t in enumerate(self.tlist):
            
            # --- 3a. Check for quench (H change) ---
            target_hz = current_hz
            if self.quench_params:
                while (quench_segment < len(self.quench_params) and 
                       t >= self.quench_params[quench_segment][0]):
                    quench_segment += 1
                
                if quench_segment < len(self.quench_params):
                    target_hz = self.quench_params[quench_segment][1]
                else:
                    target_hz = self.quench_params[-1][1]

            # --- 3b. Re-build step operator if H changed ---
            if U_step is None:
                H_changed = True
            else:
                H_changed = (target_hz != current_hz)

            if H_changed:
                if t > 0:
                    print(f"Time {t:.2f}: Updating Hamiltonian, hz = {target_hz}")
                current_hz = target_hz
                
                self.mfim.hz = current_hz
                self.mfim.get_H()
                current_H = self.mfim.H
                
                U_step = spalin.expm(-1j * current_H * self.tstep)

            # --- 3c. Calculate OTOC at time t ---
            product = self.rho @ current_A_t @ B @ current_A_t @ B
            otoc_F = np.real(product.diagonal().sum())
            self.otoc_list[i] = 1.0 - otoc_F

            # --- 3d. Evolve to next time step ---
            if i < len(self.tlist) - 1:
                current_A_t = U_step.getH() @ current_A_t @ U_step

        
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
            'Z': self.mfim.sz_list
        }

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

        U = spalin.expm(-1j * self.mfim.H * t) # Compute U(t) = exp(-i * H * t)
        U_dag = U.getH()  # Compute U^{\dagger}
        S_i_t = U_dag @ A @ U  # Compute the time-evolved operator S_i(t)
        product = self.rho @ S_i_t @ B @ S_i_t @ B  # Compute the product S_i(t) S_j S_i(t) S_j
        otoc = 1-np.real(product.diagonal().sum())   # Compute trace

        return otoc
    
    def plot_otoc(self):

        plt.figure(figsize=(8,4))
        if self.quench_params:
            label_added = False
            for (t_end, hz_val) in self.quench_params:
                if t_end < self.T:
                    if not label_added:
                        plt.axvline(x=t_end, color='red', linestyle='--', linewidth=1, label='Quench')
                        label_added = True
                    else:
                        plt.axvline(x=t_end, color='red', linestyle='--', linewidth=1)
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
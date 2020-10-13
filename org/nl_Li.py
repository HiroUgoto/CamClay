import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
import sys

class StateParameters:
    def __init__(self,strain,stress,dstrain,dstress,ef1=False,ef2=False):
        self.strain = np.copy(strain)
        self.stress = np.copy(stress)
        self.dstress = np.copy(dstress)
        self.dstrain = np.copy(dstrain)
        self.pmin = 1.0

        self.set_stress_variable()
        self.set_stress_increment()

        self.elastic_flag1 = ef1
        self.elastic_flag2 = ef2


    def set_stress_variable(self):
        self.p = (self.stress[0,0]+self.stress[1,1]+self.stress[2,2])/3.0
        self.sij = self.stress - self.p*np.eye(3)
        self.rij = self.sij / max(self.p,self.pmin)
        self.R = np.sqrt(1.5*np.power(self.rij,2).sum())     # Eq.(3)

    def set_stress_increment(self):
        stress = self.stress + self.dstress
        p = (stress[0,0]+stress[1,1]+stress[2,2])/3.0
        self.dp = p - self.p


class Li2002:
    # Defalt parameters are for Toyoura sand (Li2000)
    def __init__(self,G0=125,nu=0.25,M=1.25,c=0.75,eg=0.934,rlambdac=0.019,xi=0.7, \
                 d1=0.41,m=3.5,h1=3.15,h2=3.05,h3=2.2,n=1.1, \
                 d2=1,h4=3.5,a=1,b1=0.005,b2=2,b3=0.01):
        # Elastic parameters
        self.G0,self.nu = G0,nu
        # Critical state parameters
        self.M,self.c,self.eg,self.rlambdac,self.xi = M,c,eg,rlambdac,xi
        # parameters associated with dr-mechanisms
        self.d1,self.m,self.h1,self.h2,self.h3,self.n = d1,m,h1,h2,h3,n
        # parameters associated with dp-mechanisms
        self.d2,self.h4 = d2,h4
        # Default parameters
        self.a,self.b1,self.b2,self.b3 = a,b1,b2,b3

        # minimum epsillon
        self.eps = 1.e-6

        # stress parameters
        self.pr = 101.e3
        self.pmin = 100.0

        # stress & strain
        self.stress = np.zeros((3,3))
        self.strain = np.zeros((3,3))

        # BS parameters
        self.alpha = np.zeros((3,3))
        self.beta = 0.0
        self.H1 = 0.0
        self.H2 = 0.0

        # Accumulated index
        self.L1 = 0.0

    # -------------------------------------------------------------------------------------- #
    def vector_to_matrix(self,vec):
        mat = np.array([[vec[0],vec[3],vec[5]],
                        [vec[3],vec[1],vec[4]],
                        [vec[5],vec[4],vec[2]]])
        return mat

    def matrix_to_vector(self,mat):
        vec = np.array([mat[0,0],mat[1,1],mat[2,2],mat[0,1],mat[1,2],mat[2,0]])
        return vec

    def clear_strain(self):
        self.strain = np.zeros((3,3))

    # -------------------------------------------------------------------------------------- #
    def set_strain_variable(self,strain_mat):
        ev = strain_mat[0,0]+strain_mat[1,1]+strain_mat[2,2]
        dev_strain = strain_mat - ev/3.0 * np.eye(3)
        gamma = np.sqrt(2.0/3.0*np.power(dev_strain,2).sum())
        return ev,gamma

    def set_strain_increment(self,dstrain_mat):
        strain_mat = self.strain_mat + dstrain_mat
        ev0,gamma0 = self.set_strain_variable(self.strain_mat)
        ev,gamma = self.set_strain_variable(strain_mat)
        return ev-ev0,gamma-gamma0

    # -------------------------------------------------------------------------------------- #
    def set_stress_variable(self,stress):
        p = (stress[0,0]+stress[1,1]+stress[2,2])/3.0
        dev_stress = stress - p*np.eye(3)
        r_stress = dev_stress / max(p,self.pmin)
        R = np.sqrt(1.5*np.power(r_stress,2).sum())
        return p,R

    # -------------------------------------------------------------------------------------- #
    def Lode_angle(self,dev_stress):
        J2 = 0.5*np.power(dev_stress,2).sum()
        J3 = -np.linalg.det(dev_stress)
        if J2 == 0.0:
            s3 = 0.0
        else:
            s3 = J3/2 * (3/J2)**1.5
            s3 = max(s3,-1.0)
            s3 = min(s3,1.0)
        theta = np.arcsin(s3)/3.0
        return theta

    def g_theta(self,dev_stress):            # Eq.(7)
        theta = self.Lode_angle(dev_stress)
        st = np.sin(3*theta)
        if st == 0.0:
            g1 = self.c*(1+self.c)
            g2 = 1+self.c**2
        else:
            g1 = np.sqrt((1+self.c**2)**2 + 4*self.c*(1-self.c**2)*st) - (1+self.c**2)
            g2 = 2*(1-self.c)*st
        return g1/g2

    def dg_theta(self,theta):                 # Eq.(45)
        st = np.sin(3*theta)
        if st == 0.0:
            g1 = -self.c**2*(1-self.c)*(1+self.c)**2
            g2 = (1+self.c**2)**3
            dg = g1/g2
        else:
            g11 = self.c*(1+self.c)
            g12 = st*np.sqrt((1+self.c**2)**2+4*self.c*(1-self.c**2)*st)
            g21 = np.sqrt((1+self.c**2)**2 + 4*self.c*(1-self.c**2)*st) - (1+self.c**2)
            g22 = 2*(1-self.c)*st * st
            dg = g11/g12 - g21/g22
        return dg

    # -------------------------------------------------------------------------------------- #
    def state_parameter(self,e,p):
        psi = e - (self.eg-self.rlambdac*(max(p,self.pmin)/self.pr)**self.xi)  # Eq.(18)
        return psi

    # -------------------------------------------------------------------------------------- #
    def elastic_modulus(self,e,p):
        G = self.G0*(2.97-e)**2 / (1+e) * np.sqrt(max(p,self.pmin)*self.pr)  # Eq.(16)
        K = G*2*(1+self.nu)/(3*(1-2*self.nu))                           # Eq.(17)
        return G,K

    def elastic_stiffness(self,G):
        mu,rlambda = G,2*G*self.nu/(1-2*self.nu)
        Dijkl = np.einsum('ij,kl->ijkl',np.eye(3),np.eye(3))
        Dikjl = np.einsum('ij,kl->ikjl',np.eye(3),np.eye(3))
        Ee = rlambda*Dijkl + 2*mu*Dikjl
        return Ee

    def isotropic_compression_stiffness(self,e,p):
        G,K = self.elastic_modulus(e,p)
        fn = 2*(1+self.nu)/(3*(1-2*self.nu))
        K2 = G*fn*self.h4 / (self.h4 + np.sqrt(2/3)*fn*self.d2)   #  Elastic + Eq.(29)
        G2 = K2 / fn
        E2 = self.elastic_stiffness(G2)
        return E2

    # -------------------------------------------------------------------------------------- #
    def set_mapping_stress(self,sp):
        def mapping_r(t,rij,alpha):
            rij_bar = alpha + t*(rij-alpha)
            g_bar = self.g_theta(rij_bar)
            R_bar = np.sqrt(1.5*np.power(rij_bar,2).sum())
            return rij_bar,R_bar,g_bar

        def F1_boundary_surface(t,*args):               # Eq.(6)
            rij,alpha = args
            _,R_bar,g_bar = mapping_r(t,rij,alpha)
            return R_bar-self.H1*g_bar

        if sp.elastic_flag1:
            self.alpha = np.copy(sp.rij)
        if sp.elastic_flag2:
            self.beta = np.copy(sp.p)

        if np.linalg.norm(sp.rij-self.alpha) < 1.e-6:  # Elastic behavior
            sp.elastic_flag1 = True
        else:
            if F1_boundary_surface(1.0,sp.rij,self.alpha) > 0.0:
                sp.rij_bar,sp.R_bar,sp.g_bar = mapping_r(1.0,sp.rij,self.alpha)
                self.H1 = sp.R_bar/sp.g_bar
                sp.rho1_ratio = 1.0
            else:
                t = scipy.optimize.brentq(F1_boundary_surface,1.0,1.e6,args=(sp.rij,self.alpha))
                sp.rij_bar,sp.R_bar,sp.g_bar = mapping_r(t,sp.rij,self.alpha)
                sp.rho1_ratio = np.copy(t)      # rho1_ratio = rho1_bar / rho1

        if np.abs(sp.p-self.beta) == 0.0:  # Elastic behavior
            sp.elastic_flag2 = True
        else:
            if sp.p > self.H2:
                self.H2 = np.copy(sp.p)
            if sp.dp > 0.0:
                if sp.p <= self.beta:
                    sp.elastic_flag2 = True
                    return
                sp.p_bar = np.copy(self.H2)
            elif sp.dp < 0.0:
                if self.beta <= sp.p:
                    sp.elastic_flag2 = True
                    return
                sp.p_bar = self.pmin
            else:
                sp.elastic_flag2 = True
                return
            rho2 = np.abs(sp.p-self.beta)
            rho2_b = np.abs(sp.p_bar-self.beta)
            sp.rho2_ratio = rho2_b / rho2

    # -------------------------------------------------------------------------------------- #
    def set_parameters(self,sp):
        def accumulated_load_index(L1):         # Eq.(22)
            fL = (1-self.b3)/np.sqrt((1-L1/self.b1)**2+(L1/self.b1)/self.b2**2) + self.b3
            return fL

        def scaling_factor(e,rho1_ratio):         # Eq.(21)
            fL = accumulated_load_index(self.L1)
            r1 = (1.0/rho1_ratio)**10
            h = (self.h1-self.h2*self.e)*(r1+self.h3*fL*(1-r1))
            return h

        def plastic_modulus1(G,R_bar,g_bar,rho1_ratio,h,psi):    # Eq.(19)
            Mg_R = self.M*g_bar*np.exp(-self.n*psi) / R_bar
            Kp1 = G*h*(Mg_R*rho1_ratio - 1)
            Kp1_b = G*h*(Mg_R - 1)
            return Kp1,Kp1_b

        def dilatancy1(R,g,rho1_ratio,psi):      # Eq.(23)
            R_Mg = R / (self.M*g)
            D1 = self.d1*(np.exp(self.m*psi)*np.sqrt(rho1_ratio) - R_Mg)
            return D1

        def plastic_modulus2(G,Mg_R,rho2_ratio,sign):   #  Eq.(25)
            Kp2 = G*self.h4*Mg_R * (rho2_ratio)**self.a*sign
            if rho2_ratio == 1.0 and sign > 0.0:
                Kp2_b = np.copy(Kp2)
            else:
                Kp2_b = 0.0
            return Kp2,Kp2_b

        def dilatancy2(Mg_R,sign):               #  Eq.(27)
            if Mg_R >= 1.0:
                D2 = self.d2*(Mg_R-1.0)*sign
            else:
                D2 = 0.0
            return D2

        sp.Ge,sp.Ke = self.elastic_modulus(self.e,sp.p)
        sp.psi = self.state_parameter(self.e,sp.p)
        sp.g = self.g_theta(sp.sij)

        if sp.elastic_flag1:
            sp.Kp1_b = 0.0
            sp.D1 = 0.0
        else:
            h = scaling_factor(self.e,sp.rho1_ratio)
            sp.h = h
            sp.Kp1,sp.Kp1_b = plastic_modulus1(sp.Ge,sp.R_bar,sp.g_bar,sp.rho1_ratio,h,sp.psi)
            sp.D1 = dilatancy1(sp.R,sp.g,sp.rho1_ratio,sp.psi)

        if sp.elastic_flag2 or sp.R == 0.0:
            sp.Kp2_b = 0.0
            sp.D2 = 0.0
        else:
            sign = sp.dp/np.abs(sp.dp)
            Mg_R = self.M*sp.g/sp.R
            sp.Kp2,sp.Kp2_b = plastic_modulus2(sp.Ge,Mg_R,sp.rho2_ratio,sign)
            sp.D2 = dilatancy2(Mg_R,sign)


    # -------------------------------------------------------------------------------------- #
    def set_parameter_nm(self,sp):
        def dF1_r(r_bar,R_bar,theta_bar,g_bar,dg_bar):
            if np.abs(R_bar) < self.eps:
                return np.zeros((3,3))
            st_bar = np.sin(3*theta_bar)
            a = R_bar*g_bar + 3*R_bar*st_bar*dg_bar
            b = 9*dg_bar
            c = 1.5/(R_bar*g_bar)**2
            rr_bar = np.einsum('im,jm->ij',r_bar,r_bar)
            return (a*r_bar + b*rr_bar)*c

        if not sp.elastic_flag1:
            theta_bar = self.Lode_angle(sp.sij)
            dg_bar = self.dg_theta(theta_bar)
            dF1 = dF1_r(sp.rij_bar,sp.R_bar,theta_bar,sp.g_bar,dg_bar)
            dF1_tr = np.trace(dF1)
            nij = dF1 - np.eye(3,3)*dF1_tr/3.0
            sp.nij = nij / np.linalg.norm(nij)

        r_abs = np.linalg.norm(sp.rij)
        if r_abs == 0.0:
            sp.mij = np.zeros((3,3))
        else:
            sp.mij = sp.rij / r_abs         # mij = rij/|rij|

    # -------------------------------------------------------------------------------------- #
    def set_parameter_TZ(self,sp):
        if sp.elastic_flag1 or sp.elastic_flag2 or sp.R == 0.0:
            sp.B = 0.0
        else:
            nm = np.einsum("ij,ij",sp.nij,sp.mij)
            nr = np.einsum("ij,ij",sp.nij,sp.rij)
            Bu = 2*sp.Ge*nm - np.sqrt(2/3)*sp.Ke*sp.D2*nr
            Bd = np.sqrt(2/3)*sp.Ke*sp.D2 + sp.Kp2
            sp.B = Bu / Bd

        if sp.elastic_flag1:
            sp.Tij = np.zeros((3,3))
        else:
            nr = np.einsum("ij,ij",sp.nij,sp.rij)
            Tu = 2*sp.Ge*sp.nij - sp.Ke*(nr+sp.B)*np.eye(3)
            Td = 2*sp.Ge - np.sqrt(2/3)*sp.Ke*sp.D1*(nr+sp.B) + sp.Kp1
            sp.Tij = Tu / Td

        if sp.elastic_flag2:
            sp.Zij = np.zeros((3,3))
        else:
            Zu = sp.Ke*np.eye(3) - np.sqrt(2/3)*sp.Ke*sp.D1*sp.Tij
            if sp.R == 0.0:
                Kp2_D2 = sp.Ge*self.h4/self.d2 * sp.rho2_ratio**self.a
                Zd = np.sqrt(2/3)*sp.Ke + Kp2_D2
            else:
                Zd = np.sqrt(2/3)*sp.Ke*sp.D2 + sp.Kp2
            sp.Zij = Zu / Zd

    # -------------------------------------------------------------------------------------- #
    def set_tensor_Ep(self,sp):
        Lm0 = np.einsum('pk,ql->pqkl',np.eye(3),np.eye(3))
        if sp.elastic_flag1:
            Lm1 = np.einsum('pq,kl->pqkl',np.zeros((3,3)),np.zeros((3,3)))
        else:
            nD = sp.nij + np.sqrt(2/27)*sp.D1*np.eye(3)
            Lm1 = np.einsum("pq,kl->pqkl",nD,sp.Tij)

        if sp.elastic_flag2:
            Lm2 = np.einsum('pq,kl->pqkl',np.zeros((3,3)),np.zeros((3,3)))
        elif sp.R == 0:
            mD = np.sqrt(2/27)*np.eye(3)
            Lm2 = np.einsum("pq,kl->pqkl",mD,sp.Zij)
        else:
            mD = sp.mij + np.sqrt(2/27)*sp.D2*np.eye(3)
            Lm2 = np.einsum("pq,kl->pqkl",mD,sp.Zij)

        Lm = Lm0 - Lm1 - Lm2
        Ee = self.elastic_stiffness(sp.Ge)
        sp.Ep = np.einsum('ijpq,pqkl->ijkl',Ee,Lm)

    # -------------------------------------------------------------------------------------- #
    def check_unload(self,sp):
        self.set_mapping_stress(sp)
        self.set_parameters(sp)
        self.set_parameter_nm(sp)
        self.set_parameter_TZ(sp)

        dL1 = np.einsum("ij,ij",sp.Tij,sp.dstrain)
        if dL1 < 0.0:
            elastic_flag1 = True
            self.alpha = np.copy(sp.rij)
        else:
            elastic_flag1 = False

        dL2 = np.einsum("ij,ij",sp.Zij,sp.dstrain)
        if dL2 < 0.0:
            elastic_flag2 = True
            self.beta = np.copy(sp.p)
            print(self.beta)
        else:
            elastic_flag2 = False

        return elastic_flag1,elastic_flag2

    # -------------------------------------------------------------------------------------- #
    def update_parameters(self,sp):
        self.set_mapping_stress(sp)
        self.set_parameters(sp)
        self.set_parameter_nm(sp)
        self.set_parameter_TZ(sp)

        dL1 = np.einsum("ij,ij",sp.Tij,sp.dstrain)
        dL2 = np.einsum("ij,ij",sp.Zij,sp.dstrain)
#        print(" dL:",sp.elastic_flag1,dL1,sp.elastic_flag2,dL2)

        if not sp.elastic_flag1:
            self.L1 += dL1
            self.H1 += sp.Kp1_b*dL1 / sp.p

        if not sp.elastic_flag2:
            self.H2 += sp.Kp2_b*dL2


    # -------------------------------------------------------------------------------------- #
    def plastic_stiffness(self,sp):
        self.set_mapping_stress(sp)
        self.set_parameters(sp)
        self.set_parameter_nm(sp)
        self.set_parameter_TZ(sp)
        self.set_tensor_Ep(sp)
        return sp.Ep

    # -------------------------------------------------------------------------------------- #
    def solve_strain(self,stress_mat,E):
        b = stress_mat.flatten()
        A = np.reshape(E,(9,9))
        x = np.linalg.solve(A,b)
        strain_mat = np.reshape(x,(3,3))
        return strain_mat

    def solve_strain_with_consttain(self,strain_given,stress_given,E,deformation):
        # deformation: True => deform (stress given), False => constrain (strain given)
        d = deformation.flatten()
        A = np.reshape(E,(9,9))

        strain = np.copy(strain_given.flatten())
        strain[d] = 0.0                        # [0.0,0.0,given,...]
        stress_constrain = np.dot(A,strain)
        stress = np.copy(stress_given.flatten()) - stress_constrain

        stress_mask = stress[d]
        A_mask = A[d][:,d]
        strain_mask = np.linalg.solve(A_mask,stress_mask)

        strain[d] = strain_mask
        stress = np.dot(A,strain)

        return np.reshape(strain,(3,3)), np.reshape(stress,(3,3))

    # -------------------------------------------------------------------------------------- #
    def elastic_deformation(self,dstrain,dstress,Ge,deformation):
        Ee = self.elastic_stiffness(Ge)
        dstrain_elastic,dstress_elastic = self.solve_strain_with_consttain(dstrain,dstress,Ee,deformation)
        return dstrain_elastic,dstress_elastic

    def plastic_deformation(self,dstrain_given,dstress_given,deformation,sp0):
        ef1,ef2 = self.check_unload(sp0)
        sp = StateParameters(self.strain,self.stress,dstrain_given,dstress_given,ef1=ef1,ef2=ef2)
        Ep = self.plastic_stiffness(sp)
        dstrain_ep,dstress_ep = self.solve_strain_with_consttain(dstrain_given,dstress_given,Ep,deformation)

        sp2 = StateParameters(self.strain,self.stress,dstrain_ep,dstress_ep)
        self.update_parameters(sp2)

        dstrain = np.copy(dstrain_ep)
        dstress = np.copy(dstress_ep)
        return dstrain,dstress,sp2

    # -------------------------------------------------------------------------------------- #
    def isotropic_compression(self,e0,compression_stress,nstep=1000):
        dcp = compression_stress / nstep
        self.e = np.copy(e0)

        dstress_vec = np.array([dcp,dcp,dcp,0,0,0])
        dstress = self.vector_to_matrix(dstress_vec)

        for i in range(0,nstep):
            p,_ = self.set_stress_variable(self.stress)
            E = self.isotropic_compression_stiffness(self.e,p)
            dstrain = self.solve_strain(dstress,E)

            self.stress += dstress
            self.strain += dstrain
            ev,_ = self.set_strain_variable(self.strain)
            self.e = e0 - ev*(1+e0)

        self.clear_strain()
        
    # -------------------------------------------------------------------------------------- #
    def triaxial_compression(self,e0,compression_stress,de=0.0001,emax=0.20,print_result=False,plot=False):
        self.isotropic_compression(e0,compression_stress)
        self.e0 = np.copy(e0)
        self.e = np.copy(e0)

        p,_ = self.set_stress_variable(self.stress)
        self.beta,self.H2 = p,p

        nstep = int(emax/de)
        dstrain_vec = np.array([0.0,0.0,de,0.0,0.0,0.0])
        dstrain_input = self.vector_to_matrix(dstrain_vec)

        dstress_vec = np.array([0.0,0.0,0.0,0.0,0.0,0.0])
        dstress_input = self.vector_to_matrix(dstress_vec)

        deformation_vec = np.array([True,True,False,True,True,True],dtype=bool)
        deformation = self.vector_to_matrix(deformation_vec)

        gamma_list,R_list = [],[]
        ev_list = []
        for i in range(0,nstep):
            p,R = self.set_stress_variable(self.stress)
            dstrain,dstress = \
                self.plastic_deformation(dstrain_input,dstress_input,deformation)

            self.stress += dstress
            self.strain += dstrain

            ev,gamma = self.set_strain_variable(self.strain)
            self.e = self.e0 - ev*(1+self.e0)

            print(gamma,R,ev,p)

            gamma_list += [gamma]
            R_list += [R]
            ev_list += [ev]

        if print_result:
            print("+++ triaxial_compression +++")
            print(" e0:",self.e0)
            print("  e:",self.e)

        if plot:
            plt.figure()
            plt.plot(gamma_list,R_list)
            plt.show()

            plt.plot(gamma_list,ev_list)
            plt.show()

    # -------------------------------------------------------------------------------------- #
    def cyclic_shear_test(self,e0,compression_stress,sr=0.2,print_result=False,plot=False):
        def cycle_load(nstep,amp,i):
            tau = amp*np.sin((i+1)/nstep*2*np.pi)
            tau0 = amp*np.sin((i+0)/nstep*2*np.pi)
            return tau - tau0

        self.isotropic_compression(e0,compression_stress)
        self.e0 = np.copy(e0)
        self.e = np.copy(e0)

        p0,_ = self.set_stress_variable(self.stress)
        self.beta,self.H2 = p0,p0

        nstep = 1000
        ncycle = 10

        dstrain_vec = np.array([0.0,0.0,0.0,0.0,0.0,0.0])
        dstrain_input = self.vector_to_matrix(dstrain_vec)

        dstress_vec = np.array([0.0,0.0,0.0,0.0,0.0,0.0])
        dstress_input = self.vector_to_matrix(dstress_vec)

        deformation_vec = np.array([False,False,False,True,True,True],dtype=bool)
        deformation = self.vector_to_matrix(deformation_vec)

        sp0 = StateParameters(self.strain,self.stress,dstrain_input,dstress_input)

        gamma_list,tau_list = [],[]
        p_list = []
        step_list,ep_list = [],[]
        for ic in range(0,ncycle):
            print("N :",ic+1)
            for i in range(0,nstep):
                dtau = cycle_load(nstep,sr*p0,i)
                dstress_vec = np.array([0.0,0.0,0.0,0.0,0.0,dtau])
                dstress_input = self.vector_to_matrix(dstress_vec)

                sp = StateParameters(self.strain,self.stress,sp0.dstrain,dstress_input)

                p,R = self.set_stress_variable(self.stress)
                dstrain,dstress,sp0 = \
                    self.plastic_deformation(dstrain_input,dstress_input,deformation,sp)

                self.stress += dstress
                self.strain += dstrain

                ev,gamma = self.set_strain_variable(self.strain)
                self.e = self.e0 - ev*(1+self.e0)

                gamma_list += [self.strain[0,2]]
                tau_list += [self.stress[0,2]]
                p_list += [p]
                step_list += [ic*nstep+i]
                ep_list += [(p0-p)/p0]

        if plot:
            plt.figure()
            # plt.plot(gamma_list,tau_list)
            # plt.show()
            plt.plot(p_list,tau_list)
            plt.show()
            # plt.plot(step_list,ep_list)
            # plt.show()

# --------------------------------#
def cycle_load(nstep,amp,i):
    tau = amp*np.sin((i+1)/nstep*2*np.pi)
    tau0 = amp*np.sin((i+0)/nstep*2*np.pi)
    return tau - tau0


if __name__ == "__main__":

    Li_model = Li2002()

    emax = 0.99
    emin = 0.640
    Dr = 0.65
    e0 = emax-Dr*(emax-emin)

    horizontal_stress = 40.e3
    vertical_stress = 40.e3

    #--- initial isotropic compression ---#
    Li_model.isotropic_compression(e0,horizontal_stress)
    Li_model.e0 = np.copy(e0)
    Li_model.e = np.copy(e0)

    #--- initial triaxial compression ---#
    ds = 1.e3
    p,_ = Li_model.set_stress_variable(Li_model.stress)
    Li_model.beta,Li_model.H2 = np.copy(p),np.copy(p)

    nstep = int((vertical_stress-p)/ds)
    dstrain_vec = np.array([0.0,0.0,0.0,0.0,0.0,0.0])
    dstrain_input = Li_model.vector_to_matrix(dstrain_vec)

    dstress_vec = np.array([0.0,ds,0.0,0.0,0.0,0.0])
    dstress_input = Li_model.vector_to_matrix(dstress_vec)

    deformation_vec = np.array([True,True,True,True,True,True],dtype=bool)
    deformation = Li_model.vector_to_matrix(deformation_vec)

    sp0 = StateParameters(Li_model.strain,Li_model.stress,dstrain_input,dstress_input)

    gamma_list,R_list = [],[]
    ev_list = []
    for i in range(0,nstep):
        sp = StateParameters(Li_model.strain,Li_model.stress,sp0.dstrain,dstress_input)

        p,R = Li_model.set_stress_variable(Li_model.stress)
        dstrain,dstress,sp0 = \
            Li_model.plastic_deformation(dstrain_input,dstress_input,deformation,sp)

        Li_model.stress += dstress
        Li_model.strain += dstrain

        ev,gamma = Li_model.set_strain_variable(Li_model.strain)
        Li_model.e = Li_model.e0 - ev*(1+Li_model.e0)

        # print(gamma,R,ev,p)
        gamma_list += [gamma]
        R_list += [R]
        ev_list += [ev]

    print(Li_model.stress)
    print(Li_model.strain)

    # plt.figure()
    # plt.plot(gamma_list,R_list)
    # plt.show()
    #
    # plt.plot(gamma_list,ev_list)
    # plt.show()


    #--- cyclic shear ---#
    nstep = 500
    ncycle = 10
    sr = 0.4

    p0,_ = Li_model.set_stress_variable(Li_model.stress)
    Li_model.strain = np.zeros((3,3))

    dstrain_vec = np.array([0.0,0.0,0.0,0.0,0.0,0.0])
    dstrain_input = Li_model.vector_to_matrix(dstrain_vec)

    dstress_vec = np.array([0.0,0.0,0.0,0.0,0.0,0.0])
    dstress_input = Li_model.vector_to_matrix(dstress_vec)

    deformation_vec = np.array([False,False,False,True,False,False],dtype=bool)
    deformation = Li_model.vector_to_matrix(deformation_vec)

    sp0 = StateParameters(Li_model.strain,Li_model.stress,dstrain_input,dstress_input)
    Li_model.alpha = np.copy(sp0.rij)
    Li_model.beta,Li_model.H2 = np.copy(sp0.p),np.copy(sp0.p)

    gamma_list,tau_list = [],[]
    p_list = []
    stressxx,stressyy,stresszz=[],[],[]
    step_list,ep_list = [],[]
    for ic in range(0,ncycle):
        print("N :",ic+1)
        for i in range(0,nstep):
            dtau = cycle_load(nstep,sr*vertical_stress,i)
            if ic < 5:
                dtau *= (i+ic*nstep)/(5*nstep)

            dstress_vec = np.array([0.0,0.0,0.0,dtau,0.0,0.0])
            dstress_input = Li_model.vector_to_matrix(dstress_vec)

            sp = StateParameters(Li_model.strain,Li_model.stress,sp0.dstrain,dstress_input)

            p,R = Li_model.set_stress_variable(Li_model.stress)
            dstrain,dstress,sp0 = \
                Li_model.plastic_deformation(dstrain_input,dstress_input,deformation,sp)

            Li_model.stress += dstress
            Li_model.strain += dstrain

            ev,gamma = Li_model.set_strain_variable(Li_model.strain)
            Li_model.e = Li_model.e0 - ev*(1+Li_model.e0)

            gamma_list += [Li_model.strain[0,1]]
            tau_list += [Li_model.stress[0,1]]
            stressxx += [Li_model.stress[0,0]]
            stressyy += [Li_model.stress[1,1]]
            stresszz += [Li_model.stress[2,2]]
            p_list += [p]
            step_list += [ic*nstep+i]
            ep_list += [(p0-p)/p0]

            # print(gamma_list[-1],tau_list[-1],p_list[-1])

    plt.figure()
    plt.plot(step_list,p0-np.array(p_list))
    plt.show()

    plt.plot(step_list,tau_list)
    plt.show()

    # plt.plot(p_list,tau_list)
    # plt.show()

    plt.plot(step_list,stressxx,label="x")
    plt.plot(step_list,stressyy,label="y")
    plt.plot(step_list,stresszz,label="z")
    plt.legend()
    plt.show()

import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
import sys


class CamClay():
    def __init__(self,nu=0.3,e0=1.5,p0=100.e3,M=1.25,kappa=0.05,rlambda=0.23,Dd=0.07):
        # elastic parameters
        self.nu = nu
        # initial stress state
        self.e0,self.p0 = e0,p0
        # critical state parameters
        self.M = M
        # parameters associated with e-log_p
        self.kappa,self.rlambda = kappa,rlambda
        # parameters associated with dilatancy
        self.Dd = Dd

        # stress parameters
        self.pr = 101.e3
        self.pmin = 100.0

        # stress & strain
        self.stress = np.zeros((3,3))
        self.strain = np.zeros((3,3))

        # initial elastic modulus
        self.G,self.K = self.elastic_modulus(self.p0)

        # identity matrix
        self.I = np.eye(3)

    # -------------------------------------------------------------------------------------- #
    def set_stress_variable(self,stress):
        p = np.trace(stress)/3.0
        dev_stress = stress - p*self.I
        r_stress = dev_stress / max(p,self.pmin)
        R = np.sqrt(1.5*np.sum(r_stress**2))
        return p,R

    # -------------------------------------------------------------------------------------- #
    def set_strain_variable(self,strain_mat):
        ev = np.trace(strain_mat)
        dev_strain = strain_mat - ev/3.0 * self.I
        gamma = np.sqrt(2.0/3.0*np.sum(dev_strain**2))
        return ev,gamma

    # -------------------------------------------------------------------------------------- #
    def elastic_modulus(self,p):
        K = (1.+self.e0)/self.kappa * max(p,self.pmin)
        G = 3*(1.-2*self.nu)/(2*(1.+self.nu)) * K
        return G,K

    def elastic_stiffness(self,p):
        G,_ = self.elastic_modulus(p)
        mu,rlambda = G,2*G*self.nu/(1-2*self.nu)
        Dijkl = np.einsum('ij,kl->ijkl',self.I,self.I)
        Dikjl = np.einsum('ij,kl->ikjl',self.I,self.I)
        Ee = rlambda*Dijkl + 2*mu*Dikjl
        return Ee

    # -------------------------------------------------------------------------------------- #
    def yield_surface_p(self,p,R,evp):
        f = (self.rlambda-self.kappa)/(1.+self.e0) * np.log(p/self.p0) \
            + self.Dd*R - evp
        return f

    def yield_surface(self,stress,evp):
        p,R = self.set_stress_variable(stress)
        f = (self.rlambda-self.kappa)/(1.+self.e0) * np.log(p/self.p0) \
            + self.Dd*R - evp
        return f


    # -------------------------------------------------------------------------------------- #
    def set_parameter_n(self,stress):
        p = np.trace(stress)/3.0
        dev_stress = stress - p*self.I
        r_stress = dev_stress / max(p,self.pmin)
        R = np.sqrt(1.5*np.power(r_stress,2).sum())

        q = p*R
        beta = self.M - R

        nij = (beta/3.*self.I / max(p,self.pmin) + 1.5*r_stress / q) * self.Dd

        return nij

    def set_parameter_H(self,stress):
        p = (stress[0,0]+stress[1,1]+stress[2,2])/3.0
        Ee = self.elastic_stiffness(p)
        nij = self.set_parameter_n(stress)

        N = np.einsum('ij,ijkl->kl',nij,Ee)
        M = np.einsum('ijkl,kl->ij',Ee,nij)
        H = np.trace(nij) + np.sum(nij*M)

        return N,M,H

    def plastic_stiffness(self,stress,gamma=0.0):
        p = (stress[0,0]+stress[1,1]+stress[2,2])/3.0
        Ee = self.elastic_stiffness(p)

        N,M,H = self.set_parameter_H(stress)
        Ep = Ee - (1.0-gamma)*np.einsum('ij,kl',M,N)/H

        return Ep

    # -------------------------------------------------------------------------------------- #
    def check_unload(self,stress,dstrain):
        N,_,_ = self.set_parameter_H(stress)
        dL = np.sum(N*dstrain)
        if dL < 0.0:
            elastic_flag = True
        else:
            elastic_flag = False

        return elastic_flag

    # -------------------------------------------------------------------------------------- #
    def stress_correction(self,dstress,evp):
        def residual_function(x,stress,evp):
            p = (stress[0,0]+stress[1,1]+stress[2,2])/3.0
            stress_x = p + (1.-x)*(stress-p)
            f = self.yield_surface(stress_x,evp)
            return f**2

        stress = self.stress + dstress
        p = (stress[0,0]+stress[1,1]+stress[2,2])/3.0
        f = self.yield_surface(stress,evp)

        if f > 0.0:
            res = scipy.optimize.minimize_scalar(residual_function,args=(stress,evp))
            stress_update = p + (1.-res.x)*(stress-p)
            return stress_update - self.stress

        else:
            return dstress


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
    def elastic_deformation(self,p,dstrain,dstress,deformation):
        Ee = self.elastic_stiffness(p)
        dstrain_elastic,dstress_elastic = self.solve_strain_with_consttain(dstrain,dstress,Ee,deformation)
        return dstrain_elastic,dstress_elastic

    def plastic_deformation(self,p,dstrain_given,dstress_given,deformation):
        # 1. elastic deformation
        dstrain_elastic,dstress_elastic = self.elastic_deformation(p,dstrain_given,dstress_given,deformation)
        stress = self.stress + dstress_elastic
        evp = np.copy(self.evp)

        # 2. check yield surface
        f = self.yield_surface(stress,evp)
        if f < 0.0:
            return dstrain_elastic,dstress_elastic,0.0

        # 3. check unloading
        if self.check_unload(stress,dstrain_elastic):
            return dstrain_elastic,dstress_elastic,0.0

        # 4. plastic loading
        E = self.plastic_stiffness(stress)
        dstrain_ep,dstress_ep = self.solve_strain_with_consttain(dstrain_given,dstress_given,E,deformation)
        devp = np.trace(dstrain_ep - dstrain_elastic)

        return dstrain_ep,dstress_ep,devp

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
    def isotropic_compression(self,compression_stress,nstep=1000):
        dcp = (compression_stress - self.p0) / nstep
        dstress = np.diag(np.array([dcp,dcp,dcp]))
        p = self.p0

        self.e = np.copy(self.e0)
        ev,evp = 0.0,0.0

        for i in range(0,nstep):
            deve = self.kappa/(1.+self.e0)/p * dcp
            if deve < 0.0:
                devp = 0.
            else:
                if self.yield_surface_p(p,0.0,evp) < 0.0:
                    devp = 0.
                else:
                    devp = (self.rlambda-self.kappa)/(1.+self.e0) / p * dcp

            p += dcp
            evp += devp
            ev += deve + devp
            e = self.e0 - ev*(1+self.e0)

        self.stress = np.diag(np.array([p,p,p]))
        self.strain = np.zeros((3,3))
        self.evp = evp
        self.e = self.e0 - ev*(1+self.e0)
        self.f0 = self.yield_surface(self.stress,self.evp)

        print(" compression stress [kPa]: ", compression_stress*1.e-3)
        print(" void ratio e: ", self.e)

    # -------------------------------------------------------------------------------------- #
    def simple_shear(self,back_pressure,gamma_max=0.01,dg=0.0001,print_result=False,plot=False):
        print("+++ initial compression +++")
        self.isotropic_compression(back_pressure)
        self.e_init = np.copy(self.e)

        print("+++ simple shear +++")
        nstep = int(gamma_max/dg)
        dstrain_vec = np.array([0.0,0.0,0.0,0.0,0.0,dg])
        dstrain_input = self.vector_to_matrix(dstrain_vec)

        dstress_vec = np.array([0.0,0.0,0.0,0.0,0.0,0.0])
        dstress_input = self.vector_to_matrix(dstress_vec)

        deformation_vec = np.array([False,False,False,False,False,False],dtype=bool)
        deformation = self.vector_to_matrix(deformation_vec)

        gamma_list,q_list = [],[]
        ev_list = []
        for i in range(0,nstep):
            p,R = self.set_stress_variable(self.stress)

            dstrain,dstress,devp = self.plastic_deformation(p,dstrain_input,dstress_input,deformation)

            self.stress += dstress
            self.strain += dstrain
            self.evp += devp

            p,R = self.set_stress_variable(self.stress)
            ev,gamma = self.set_strain_variable(self.strain)
            self.e = self.e_init - ev*(1+self.e0)
            self.f0 = self.yield_surface(self.stress,self.evp)

            gamma_list += [gamma]
            q_list += [R*p]
            ev_list += [ev]

            # print(gamma,R,ev,p,self.f0)

        if print_result:
            # print("+++ triaxial_compression +++")
            print(" e0:",self.e_init)
            print("  e:",self.e)
            print(self.stress)
            print(self.strain)

        if plot:
            plt.figure()
            plt.plot(gamma_list,q_list)
            plt.show()



    # -------------------------------------------------------------------------------------- #
    def triaxial_compression(self,compression_stress,de=0.0002,emax=1.00,print_result=False,plot=False):
        print("+++ initial compression +++")
        self.isotropic_compression(compression_stress)
        self.e_init = np.copy(self.e)

        print("+++ triaxial compression +++")
        nstep = int(emax/de)
        dstrain_vec = np.array([0.0,0.0,de,0.0,0.0,0.0])
        dstrain_input = self.vector_to_matrix(dstrain_vec)

        dstress_vec = np.array([0.0,0.0,0.0,0.0,0.0,0.0])
        dstress_input = self.vector_to_matrix(dstress_vec)

        deformation_vec = np.array([True,True,False,True,True,True],dtype=bool)
        deformation = self.vector_to_matrix(deformation_vec)

        gamma_list,q_list = [],[]
        ev_list = []
        for i in range(0,nstep):
            p,R = self.set_stress_variable(self.stress)

            dstrain,dstress,devp = self.plastic_deformation(p,dstrain_input,dstress_input,deformation)

            self.stress += dstress
            self.strain += dstrain
            self.evp += devp

            p,R = self.set_stress_variable(self.stress)
            ev,gamma = self.set_strain_variable(self.strain)
            self.e = self.e_init - ev*(1+self.e0)
            self.f0 = self.yield_surface(self.stress,self.evp)

            print(gamma,R,ev,p,self.f0)

            gamma_list += [gamma]
            q_list += [R*p]
            ev_list += [ev]

        if print_result:
            # print("+++ triaxial_compression +++")
            print(" e0:",self.e_init)
            print("  e:",self.e)
            print(self.stress)
            print(self.strain)

        if plot:
            plt.figure()
            plt.plot(gamma_list,q_list)
            plt.show()

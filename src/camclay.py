import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
import sys

import state_param

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

    # -------------------------------------------------------------------------------------- #
    def set_stress_variable(self,stress):
        p = (stress[0,0]+stress[1,1]+stress[2,2])/3.0
        dev_stress = stress - p*np.eye(3)
        r_stress = dev_stress / max(p,self.pmin)
        R = np.sqrt(1.5*np.power(r_stress,2).sum())
        return p,R

    # -------------------------------------------------------------------------------------- #
    def set_strain_variable(self,strain_mat):
        ev = strain_mat[0,0]+strain_mat[1,1]+strain_mat[2,2]
        dev_strain = strain_mat - ev/3.0 * np.eye(3)
        gamma = np.sqrt(2.0/3.0*np.power(dev_strain,2).sum())
        return ev,gamma

    # -------------------------------------------------------------------------------------- #
    def elastic_modulus(self,p):
        K = (1.+self.e0)/self.kappa * max(p,self.pmin)
        G = 3*(1.-2*self.nu)/(2*(1.+self.nu)) * K
        return G,K

    def elastic_stiffness(self,G):
        mu,rlambda = G,2*G*self.nu/(1-2*self.nu)
        Dijkl = np.einsum('ij,kl->ijkl',np.eye(3),np.eye(3))
        Dikjl = np.einsum('ij,kl->ikjl',np.eye(3),np.eye(3))
        Ee = rlambda*Dijkl + 2*mu*Dikjl
        return Ee

    # -------------------------------------------------------------------------------------- #
    def yield_surface(self,p,R,evp):
        f = (self.rlambda-self.kappa)/(1.+self.e0) * np.log(p/self.p0) \
            + self.Dd*R - evp
        return f

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
        self.e = np.copy(self.e0)

        dstress_vec = np.array([dcp,dcp,dcp,0,0,0])
        dstress = self.vector_to_matrix(dstress_vec)

        p = self.p0
        ev,evp = 0.0,0.0

        for i in range(0,nstep):
            if self.yield_surface(p,0.0,evp) < 0.0:
                devp = 0.
                deve = self.kappa/(1.+self.e0) / p * dcp
            else:
                devp = (self.rlambda-self.kappa)/(1.+self.e0) / p * dcp
                deve = self.kappa/(1.+self.e0) / p * dcp

            p += dcp
            evp += devp
            ev += deve + devp

        self.stress = np.diag(np.array([dcp,dcp,dcp]))
        self.strain = np.zeros((3,3))
        self.evp = evp
        self.e = self.e0 - ev*(1+self.e0)


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

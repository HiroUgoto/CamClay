import numpy as np

class StateParameters():
    def __init__(self,strain=np.zeros((3,3)),stress=np.zeros((3,3)), \
                      dstrain=np.zeros((3,3)),dstress=np.zeros((3,3)), \
                      ef1=False,ef2=False):
        self.strain = np.copy(strain)
        self.stress = np.copy(stress)
        self.dstress = np.copy(dstress)
        self.dstrain = np.copy(dstrain)
        self.pmin = 100.0

        self.set_stress_variable()
        self.set_stress_increment()

        self.elastic_flag1 = ef1
        self.elastic_flag2 = ef2

    # -------------------------------------------------------------------------------------- #
    def set_stress_variable(self):
        self.p = (self.stress[0,0]+self.stress[1,1]+self.stress[2,2])/3.0
        self.sij = self.stress - self.p*np.eye(3)
        self.rij = self.sij / max(self.p,self.pmin)
        self.R = np.sqrt(1.5*np.power(self.rij,2).sum())

    def set_stress_increment(self):
        stress = self.stress + self.dstress
        p = (stress[0,0]+stress[1,1]+stress[2,2])/3.0
        self.dp = p - self.p

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

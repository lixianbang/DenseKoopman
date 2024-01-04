import numpy as np
from Hankel_DMD import H_DMD

def GenerateKoopmanModes(Data, delay,delt, mode1, mode2):

    NxLn, NtLn = Data.shape[0], Data.shape[1]
    Avg = np.mean(Data, 1)

    for i in range(Data.shape[0]):
        for j in range(Data.shape[1]):
            Data[i][j] -= Avg[i]

    eigval, Modes1, bo, _, _, _ = H_DMD(Data, delay)
    omega = np.log(np.diag(eigval)) / delt
    #取复数的虚部
    Freal = np.imag(omega) / (2 * np.pi)
    #排序
    # T = np.sort((1/Freal)/60)
    # T = T[::-1]
    # T = T.reshape(-1, 1)
    Im = np.argsort((1./Freal)/60)
    Im = Im[::-1]
    Im = Im.reshape(-1, 1)

    #排序
    omega = omega[Im]
    Modes1 = Modes1[:, Im]
    Modes1 = np.reshape(Modes1, (Modes1.shape[0], Modes1.shape[1]))
    bo = bo[Im]

    time = np.linspace(0, NtLn-1, NtLn)
    time = time * delt

    Psi = np.zeros((NxLn, NtLn, mode2 - mode1 + 1), dtype=complex)
    for i in range(mode2 - mode1 + 1):
        psi = np.zeros((1, NtLn), dtype=complex)
        omeganow = omega[i]
        bnow = bo[i]

        for t in range(len(time)):
            psi[:, t] = np.exp(omeganow * time[t]) * bnow

        Modes1_ = np.zeros((NxLn, 1), dtype=complex)
        for m in range(NxLn):
            Modes1_[m][0] = Modes1[m][i]

        psi = Modes1_ * psi
        Psi[:, :, i] = psi

    return Psi
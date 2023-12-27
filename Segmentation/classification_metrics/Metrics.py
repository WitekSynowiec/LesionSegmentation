class Metrics:
    def __init__(self):
        self.__E_ppv = 0.0
        self.__E_acc = 0.0
        self.__E_bacc = 0.0
        self.__E_tpr = 0.0
        self.__E_tnr = 0.0
        self.__E_f05 = 0.0
        self.__E_f1 = 0.0
        self.__E_f2 = 0.0
        self.__E_k = 0.0
        self.__E_mcc = 0.0
        self.__E_jacc = 0.0

        self.__counter = 0

    def update(self, confusion_matrix):
        self.__counter += 1
        tn = confusion_matrix[0, 0]
        fp = confusion_matrix[0, 1]
        fn = confusion_matrix[1, 0]
        tp = confusion_matrix[1, 1]

        self.__E_ppv += tp / (tp + fp) if tp + fp != 0 else 0
        self.__E_acc += (tp + tn) / (tp + fp + tn + fn)
        self.__E_bacc += (tp / (tp + fn) + tn / (tn + fp)) / 2 if (tp + fn) != 0 and (tn + fp) != 0 else 0
        self.__E_tpr += tp / (tp + fn) if tp + fn != 0 else 0
        self.__E_tnr += tn / (tn + fp) if tn + fp != 0 else 0
        self.__E_f05 += (1+0.5**(1.0/2))*tp / ((1+0.5**(1.0/2))*tp + fp + 0.5**(1.0/2)*fn) if (2*tp + fp + fn) != 0 else 0
        self.__E_f1 += 2*tp / (2*tp + fp + fn) if (2*tp + fp + fn) != 0 else 0
        self.__E_f2 += (1+2**(1.0/2))*tp / ((1+2**(1.0/2))*tp + fp + 2**(1.0/2)*fn) if (2*tp + fp + fn) != 0 else 0
        self.__E_k += 2 * (tp * tn - fn * fp) / ((tp + fp) * (fp + tn) + (tp + fn) * (fn + tn)) if (tp + fp) * (fp + tn) + (tp + fn) * (fn + tn) != 0 else 0
        self.__E_mcc += (tp * tn - fp * fn) / (((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** (1.0 / 2)) if tp + fp != 0 and tp + fn != 0 and tn + fp != 0 and tn + fn != 0 else 0
        self.__E_jacc += tp / (tp + fp + fn) if (tp + fp + fn) else 0

    def get_counter(self):
        return self.__counter

    @property
    def ppv(self):
        return self.__E_ppv / self.__counter

    @property
    def acc(self):
        return self.__E_acc / self.__counter

    @property
    def bacc(self):
        return self.__E_bacc / self.__counter

    @property
    def tpr(self):
        return self.__E_tpr / self.__counter

    @property
    def tnr(self):
        return self.__E_tnr / self.__counter

    @property
    def f1(self):
        return self.__E_f1 / self.__counter

    @property
    def f05(self):
        return self.__E_f05 / self.__counter

    @property
    def f2(self):
        return self.__E_f2 / self.__counter

    @property
    def k(self):
        return self.__E_k / self.__counter

    @property
    def mcc(self):
        return self.__E_mcc / self.__counter

    @property
    def jacc(self):
        return self.__E_jacc / self.__counter


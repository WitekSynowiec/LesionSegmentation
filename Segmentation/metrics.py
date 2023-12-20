import torch

import torchmetrics.classification as metrics

Tensor = torch.Tensor


class BinaryMetrics:
    def __init__(self, threshold: float = .5):
        self.__beta = 1.0
        self.__threshold = threshold
        self.device = torch.device("cuda")

        self.__accuracy = metrics.BinaryAccuracy(threshold=self.__threshold).to(self.device)
        self.__auroc = metrics.BinaryAUROC().to(self.device)
        self.__bce_with_logits = torch.nn.BCELoss()
        self.__calibration_error = metrics.BinaryCalibrationError().to(self.device)
        self.__cohen_kappa = lambda tn, fp, fn, tp, alpha: (2 * (tp * tn - fn * fp) + alpha) / (
                    (tp + fp) * (fp + tn) + (tp + fn) * (fn + tn) + alpha)
        self.__confusion_matrix = metrics.BinaryConfusionMatrix(threshold=self.__threshold).to(self.device)
        self.__f1score = metrics.BinaryF1Score(threshold=self.__threshold).to(self.device)
        self.__f_beta_score = metrics.BinaryFBetaScore(beta=self.__beta, threshold=self.__threshold).to(self.device)
        self.__hamming_distance = metrics.BinaryHammingDistance(threshold=self.__threshold).to(self.device)
        self.__hinge_loss = metrics.BinaryHingeLoss().to(self.device)
        self.__jaccard_index = lambda tn, fp, fn, tp, alpha: (tp + alpha) / (tp + fp + fn + alpha)
        self.__matthew_correlation_coeff = metrics.BinaryMatthewsCorrCoef(threshold=self.__threshold).to(self.device)
        self.__mse = torch.nn.MSELoss()
        self.__precision = metrics.BinaryPrecision(threshold=self.__threshold).to(self.device)
        self.__p4score = lambda tn, fp, fn, tp, alpha: (4 * tp * tn + alpha) / (
                    4 * tp * tn + (tp + tn) * (fp + fn) + alpha)
        self.__recall = metrics.BinaryRecall(threshold=self.__threshold).to(self.device)
        self.__specificity = metrics.BinarySpecificity(threshold=self.__threshold).to(self.device)

        self.__aggregated_accuracy = None
        self.__aggregated_auroc = None
        self.__aggregated_bce_with_logits = None
        self.__aggregated_calibration_error = None
        self.__aggregated_cohen_kappa = None
        self.__aggregated_confusion_matrix = None
        self.__aggregated_dice = None
        self.__aggregated_f1score = None
        self.__aggregated_f_beta_score = None
        self.__aggregated_hamming_distance = None
        self.__aggregated_hinge_loss = None
        self.__aggregated_jaccard_index = None
        self.__aggregated_matthew_correlation_coeff = None
        self.__aggregated_mse = None
        self.__aggregated_precision = None
        self.__aggregated_p4score = None
        self.__aggregated_recall = None
        self.__aggregated_specificity = None

        self._c = 0

    def set_beta(self, beta: float):
        self.__beta = beta

    def set_device(self, device: torch.device):
        self.device = device

    """
    Updates classification_metrics by a new Tensor.
    """

    def update(self, prediction: Tensor, target: Tensor):
        self._c += 1
        alpha = 1e-8
        threshold = 0.6

        prediction.to(self.device)
        target.to(self.device)

        prediction = (prediction > threshold).float()

        prediction = torch.flatten(input=prediction, start_dim=0)
        target = torch.flatten(input=target, start_dim=0)

        confusion_matrix = self.__confusion_matrix(prediction, target)
        if self.__aggregated_confusion_matrix is not None:
            self.__aggregated_confusion_matrix += confusion_matrix
        else:
            self.__aggregated_confusion_matrix = confusion_matrix

        tn = confusion_matrix[0, 0]
        fp = confusion_matrix[0, 1]
        fn = confusion_matrix[1, 0]
        tp = confusion_matrix[1, 1]

        if self.__aggregated_accuracy is not None:
            self.__aggregated_accuracy += self.__accuracy(prediction, target)
        else:
            self.__aggregated_accuracy = self.__accuracy(prediction, target)

        if self.__aggregated_auroc is not None:
            self.__aggregated_auroc += self.__auroc(prediction, target)
        else:
            self.__aggregated_auroc = self.__auroc(prediction, target)

        if self.__aggregated_bce_with_logits is not None:
            self.__aggregated_bce_with_logits += self.__bce_with_logits(prediction, target)
        else:
            self.__aggregated_bce_with_logits = self.__bce_with_logits(prediction, target)

        if self.__aggregated_calibration_error is not None:
            self.__aggregated_calibration_error += self.__calibration_error(prediction, target)
        else:
            self.__aggregated_calibration_error = self.__calibration_error(prediction, target)

        if self.__aggregated_cohen_kappa is not None:
            self.__aggregated_cohen_kappa += self.__cohen_kappa(tn, fp, fn, tp, alpha)
        else:
            self.__aggregated_cohen_kappa = self.__cohen_kappa(tn, fp, fn, tp, alpha)

        if self.__aggregated_f1score is not None:
            self.__aggregated_f1score += self.__f1score(prediction, target)
        else:
            self.__aggregated_f1score = self.__f1score(prediction, target)

        if self.__aggregated_f_beta_score is not None:
            self.__aggregated_f_beta_score += self.__f_beta_score(prediction, target)
        else:
            self.__aggregated_f_beta_score = self.__f_beta_score(prediction, target)

        if self.__aggregated_hamming_distance is not None:
            self.__aggregated_hamming_distance += self.__hamming_distance(prediction, target)
        else:
            self.__aggregated_hamming_distance = self.__hamming_distance(prediction, target)

        if self.__aggregated_hinge_loss is not None:
            self.__aggregated_hinge_loss += self.__hinge_loss(prediction, target)
        else:
            self.__aggregated_hinge_loss = self.__hinge_loss(prediction, target)

        if self.__aggregated_jaccard_index is not None:
            self.__aggregated_jaccard_index += self.__jaccard_index(tn, fp, fn, tp, alpha)
        else:
            self.__aggregated_jaccard_index = self.__jaccard_index(tn, fp, fn, tp, alpha)

        if self.__aggregated_matthew_correlation_coeff is not None:
            self.__aggregated_matthew_correlation_coeff += self.__matthew_correlation_coeff(prediction, target)
        else:
            self.__aggregated_matthew_correlation_coeff = self.__matthew_correlation_coeff(prediction, target)

        if self.__aggregated_mse is not None:
            self.__aggregated_mse += self.__mse(prediction, target)
        else:
            self.__aggregated_mse = self.__mse(prediction, target)

        if self.__aggregated_precision is not None:
            self.__aggregated_precision += self.__precision(prediction, target)
        else:
            self.__aggregated_precision = self.__precision(prediction, target)

        if self.__aggregated_p4score is not None:
            self.__aggregated_p4score += self.__p4score(tn, fp, fn, tp, alpha)
        else:
            self.__aggregated_p4score = self.__p4score(tn, fp, fn, tp, alpha)

        if self.__aggregated_recall is not None:
            self.__aggregated_recall += self.__recall(prediction, target)
        else:
            self.__aggregated_recall = self.__recall(prediction, target)

        if self.__aggregated_specificity is not None:
            self.__aggregated_specificity += self.__specificity(prediction, target)
        else:
            self.__aggregated_specificity = self.__specificity(prediction, target)

    @property
    def accuracy(self):
        return self.__aggregated_accuracy / self._c

    @property
    def auroc(self):
        return self.__aggregated_auroc / self._c

    @property
    def bce_with_logits(self):
        return self.__aggregated_bce_with_logits / self._c

    @property
    def calibration_error(self):
        return self.__aggregated_calibration_error / self._c

    @property
    def cohen_kappa(self):
        return self.__aggregated_cohen_kappa / self._c

    @property
    def confusion_matrix(self):
        return self.__aggregated_confusion_matrix / self._c

    @property
    def f1score(self):
        return self.__aggregated_f1score / self._c

    @property
    def f_beta_score(self):
        return self.__aggregated_f_beta_score / self._c

    @property
    def hamming_distance(self):
        return self.__aggregated_hamming_distance / self._c

    @property
    def hinge_loss(self):
        return self.__aggregated_hinge_loss / self._c

    @property
    def jaccard_index(self):
        return self.__aggregated_jaccard_index / self._c

    @property
    def matthew_correlation_coeff(self):
        return self.__aggregated_matthew_correlation_coeff / self._c

    @property
    def mse(self):
        return self.__aggregated_mse / self._c

    @property
    def precision(self):
        return self.__aggregated_precision / self._c

    @property
    def p4score(self):
        return self.__aggregated_p4score / self._c

    @property
    def recall(self):
        return self.__aggregated_recall / self._c

    @property
    def specificity(self):
        return self.__aggregated_specificity / self._c

    def print_metrics(self):
        print("ACC: {}".format(self.accuracy))
        print("AUROC: {}".format(self.auroc))
        print("BCE: {}".format(self.bce_with_logits))
        print("CE: {}".format(self.calibration_error))
        print("CK: {}".format(self.cohen_kappa))
        print("CM: {}".format(self.confusion_matrix))
        print("F1: {}".format(self.f1score))
        print("FB: {}".format(self.f_beta_score))
        print("HAMc: {}".format(self.hamming_distance))
        print("HIN: {}".format(self.hinge_loss))
        print("JAC: {}".format(self.jaccard_index))
        print("MCC: {}".format(self.matthew_correlation_coeff))
        print("MSE: {}".format(self.mse))
        print("PRE: {}".format(self.precision))
        print("P4: {}".format(self.p4score))
        print("REC: {}".format(self.recall))
        print("SPE: {}".format(self.specificity))

    def __call__(self):
        if self._c == 0:
            return None
        else:
            return {
                "Accuracy": self.__aggregated_accuracy / self._c,
                "AUROC": self.__aggregated_auroc / self._c,
                "BCE with logits": self.__aggregated_bce_with_logits / self._c,
                "Calibration error": self.__aggregated_calibration_error / self._c,
                "Cohen kappa": self.__aggregated_cohen_kappa / self._c,
                "Confusion matrix": self.__aggregated_confusion_matrix / self._c,
                "F1 score": self.__aggregated_f1score / self._c,
                "F beta score": self.__aggregated_f_beta_score / self._c,
                "Hamming distance": self.__aggregated_hamming_distance / self._c,
                "Hinge loss": self.__aggregated_hinge_loss / self._c,
                "Jaccard index": self.__aggregated_jaccard_index / self._c,
                "MCC": self.__aggregated_matthew_correlation_coeff / self._c,
                "MSE": self.__aggregated_mse / self._c,
                "Precision": self.__aggregated_precision / self._c,
                "P4 score": self.__aggregated_p4score / self._c,
                "Recall": self.__aggregated_recall / self._c,
                "Specificity": self.__aggregated_specificity / self._c,
            }

if __name__ == "__main__":
    target = torch.tensor([0, 0, 0, 0, 0, 0])
    preds = torch.tensor([0, 0, 0, 0, 0, 0])
    metric = metrics.BinaryPrecision()
    print(metric(preds, target))
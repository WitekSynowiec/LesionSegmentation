import json
import os

import torch
from torch.utils.data import DataLoader
from torchmetrics.classification import BinaryConfusionMatrix, BinaryAUROC
from torch import nn
from Segmentation.classification_metrics.Metrics import Metrics


class MetricsHandler:
    def __init__(self, loader: DataLoader, model: nn.Module, thresholds: list, device=torch.device("cuda")):
        self.__loader = loader
        self.__device = device
        self.__model = model
        self.__thresholds = thresholds

    def evaluate(self):
        self.__model.eval()
        self.__model.to(device=self.__device)

        threshold_metric_dict = dict()
        for threshold in self.__thresholds:
            metric = Metrics()
            threshold_metric_dict.update({threshold: metric})

        b_auroc = BinaryAUROC()
        b_auroc_sum = 0.0
        eval_number = 0
        with torch.no_grad():
            for x, target in self.__loader:
                x = x.to(self.__device)
                target = target.to(self.__device)
                self.__model = self.__model.to(self.__device)
                prediction = torch.sigmoid(self.__model(x)).to(self.__device)
                prediction = torch.flatten(input=prediction, start_dim=0)
                target = torch.flatten(input=target, start_dim=0)

                for threshold, metric in threshold_metric_dict.items():
                    bcm = BinaryConfusionMatrix(threshold=threshold).to(self.__device)
                    cm = bcm(prediction, target)

                    threshold_metric_dict[threshold].update(cm.cpu().detach().numpy())
                    eval_number = threshold_metric_dict[threshold].get_counter()
                b_auroc_sum = b_auroc_sum + b_auroc(prediction, target).cpu().detach().numpy()

        return threshold_metric_dict, b_auroc_sum / eval_number

    def append_metrics(self, to_append : dict, new_elements : dict):
        for threshold, metric in new_elements.items():
            # to_append[key].append(value.cpu().detach().numpy().tolist())
            to_append[threshold].append(metric)
        return to_append

    def save_metrics(self, metrics: dict, auroc : list,  path: os.path):
        print("=> Saving classification_metrics")
        for key, threshold_metrics in metrics.items():
            ppv = []
            acc = []
            bacc = []
            tpr = []
            tnr = []
            f1 = []
            f05 = []
            f2 = []
            k = []
            mcc = []
            jacc = []

            for epoch_metric in threshold_metrics:
                ppv.append(epoch_metric.ppv)
                acc.append(epoch_metric.acc)
                bacc.append(epoch_metric.bacc)
                tpr.append(epoch_metric.tpr)
                tnr.append(epoch_metric.tnr)
                f1.append(epoch_metric.f1)
                f05.append(epoch_metric.f05)
                f2.append(epoch_metric.f2)
                k.append(epoch_metric.k)
                mcc.append(epoch_metric.mcc)
                jacc.append(epoch_metric.jacc)

            metrics_dict = {
                "ppv" : ppv,
                "acc" : acc,
                "bacc" : bacc,
                "tpr" : tpr,
                "tnr" : tnr,
                "f1" : f1,
                "f05" : f05,
                "f2" : f2,
                "k" : k,
                "mcc" : mcc,
                "jacc" : jacc,
            }
            for metric_name, metric_value in metrics_dict.items():
                path_name = os.path.join(path, "metrics", str(key))
                os.makedirs(path_name, exist_ok=True)
                with open(os.path.join(path_name, '{}.json'.format(metric_name)), 'w') as fp:
                    json.dump(metric_value, fp)

            with open(os.path.join(path, "metrics" ,'auroc.json'), 'w') as fp:
                json.dump(auroc, fp)

            # with open(os.path.join(path, "metrics" ,str(key), '{}.json'.format("ppv")), 'w') as fp:
            #     json.dump(ppv, fp)
            # with open(os.path.join(path, "metrics" ,str(key), '{}.json'.format("acc")), 'w') as fp:
            #     json.dump(acc, fp)
            # with open(os.path.join(path, "metrics" ,str(key), '{}.json'.format("bacc")), 'w') as fp:
            #     json.dump(bacc, fp)
            # with open(os.path.join(path, "metrics" ,str(key), '{}.json'.format("tpr")), 'w') as fp:
            #                 json.dump(tpr, fp)
            # with open(os.path.join(path, "metrics" ,str(key), '{}.json'.format("tnr")), 'w') as fp:
            #     json.dump(tnr, fp)
            # with open(os.path.join(path, "metrics" ,str(key), '{}.json'.format("f1")), 'w') as fp:
            #                 json.dump(f1, fp)
            # with open(os.path.join(path, "metrics" ,str(key), '{}.json'.format("f05")), 'w') as fp:
            #     json.dump(f05, fp)
            # with open(os.path.join(path, "metrics" ,str(key), '{}.json'.format("f2")), 'w') as fp:
            #                 json.dump(f2, fp)
            # with open(os.path.join(path, "metrics" ,str(key), '{}.json'.format("k")), 'w') as fp:
            #     json.dump(k, fp)
            # with open(os.path.join(path, "metrics" ,str(key), '{}.json'.format("mcc")), 'w') as fp:
            #                 json.dump(mcc, fp)
            # with open(os.path.join(path, "metrics" ,str(key), '{}.json'.format("jacc")), 'w') as fp:
            #     json.dump(jacc, fp)



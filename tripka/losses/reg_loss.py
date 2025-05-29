# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from unicore import metrics
from unicore.losses import UnicoreLoss, register_loss
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, f1_score
import torch.cuda.amp as amp
import torch.nn as nn

def pad_and_concat_2d(tensor1, tensor2):
    n1, m1 = tensor1.shape
    n2, m2 = tensor2.shape
    max_m = max(m1, m2)

    tensor1_padded = F.pad(tensor1, (0, max_m - m1), value=0)
    tensor2_padded = F.pad(tensor2, (0, max_m - m2), value=0)

    return torch.cat([tensor1_padded, tensor2_padded], dim=0)

def pad_and_concat_3d(tensor1, tensor2):
    if len(tensor1.shape) == 3:
        n1, m1, _ = tensor1.shape
        n2, m2, _= tensor2.shape
        max_m = max(m1, m2)

        tensor1_padded = F.pad(tensor1, (0, max_m - m1, 0, max_m - m1), value=-100)
        tensor2_padded = F.pad(tensor2, (0, max_m - m2, 0, max_m - m2), value=-100)
    elif len(tensor1.shape) == 4:
        b1, n1, n1_, c1 = tensor1.shape
        b2, n2, n2_, c2 = tensor2.shape

        assert c1 == c2 == 3, "Both tensors must have 3 channels"

        max_n = max(n1, n2)

        tensor1_padded = F.pad(tensor1, (0, 0, 0, max_n - n1, 0, max_n - n1), value=-100)
        tensor2_padded = F.pad(tensor2, (0, 0, 0, max_n - n2, 0, max_n - n2), value=-100)

    return torch.cat([tensor1_padded, tensor2_padded], dim=0)

@register_loss("finetune_mse")
class FinetuneMSELoss(UnicoreLoss):
    def __init__(self, task):
        super().__init__(task)

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output_a, batch_a = model(
            sample["net_input_a"],
            classification_head_name=self.args.classification_head_name,
            features_only = True,
        )
        net_output_b, batch_b = model(
            sample["net_input_b"],
            classification_head_name=self.args.classification_head_name,
            features_only = True,
        )

        loss, predict = self.compute_loss(model, net_output_a, net_output_b, batch_a, batch_b, sample, reduce=reduce)
        sample_size = sample["target"]["finetune_target"].size(0)
        if not self.training:
            if self.task.mean and self.task.std:
                targets_mean = torch.tensor(self.task.mean, device=predict.device)
                targets_std = torch.tensor(self.task.std, device=predict.device)
                predict = predict * targets_std + targets_mean
            logging_output = {
                "loss": loss.data,
                "predict": predict.view(-1, self.args.num_classes).data,
                "target": sample["target"]["finetune_target"]
                .view(-1, self.args.num_classes)
                .data,
                "smi_name": sample["id"],
                "sample_size": sample_size,
                "num_task": self.args.num_classes,
                "conf_size": self.args.conf_size,
                "bsz": sample["target"]["finetune_target"].size(0),
            }
        else:
            logging_output = {
                "loss": loss.data,
                "sample_size": sample_size,
                "bsz": sample["target"]["finetune_target"].size(0),
            }
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output_a, net_output_b, batch_a, batch_b, sample, reduce=True):
        free_energy_a = net_output_a.view(-1, self.args.num_classes).float()
        # print(f"free energy_a: {free_energy_a}")
        free_energy_b = net_output_b.view(-1, self.args.num_classes).float()
        # print(f"free energy_b: {free_energy_b}")
        if not self.training:
            def compute_agg_free_energy(free_energy, batch):
                split_tensor_list = torch.split(free_energy, self.args.conf_size, dim=0)
                mean_tensor_list = [torch.mean(x, dim=0, keepdim=True) for x in split_tensor_list]
                agg_free_energy = torch.cat(mean_tensor_list, dim=0)
                agg_batch = [x//self.args.conf_size for x in batch]
                return agg_free_energy, agg_batch
            free_energy_a, batch_a = compute_agg_free_energy(free_energy_a, batch_a)
            free_energy_b, batch_b = compute_agg_free_energy(free_energy_b, batch_b)

        free_energy_a_padded = torch.nn.utils.rnn.pad_sequence(
            torch.split(free_energy_a, batch_a),
            padding_value=float("inf")
        )
        free_energy_b_padded = torch.nn.utils.rnn.pad_sequence(
            torch.split(free_energy_b, batch_b),
            padding_value=float("inf")
        )
        predicts = (
            torch.logsumexp(-free_energy_a_padded, dim=0)-
            torch.logsumexp(-free_energy_b_padded, dim=0)
        ) /  torch.log(torch.tensor([10.0])).item()

        targets = (
            sample["target"]["finetune_target"].view(-1, self.args.num_classes).float()
        )
        if self.task.mean and self.task.std:
            targets_mean = torch.tensor(self.task.mean, device=targets.device)
            targets_std = torch.tensor(self.task.std, device=targets.device)
            targets = (targets - targets_mean) / targets_std
        loss = F.mse_loss(
            predicts,
            targets,
            reduction="sum" if reduce else "none",
        )
        return loss, predicts

    @staticmethod
    def reduce_metrics(logging_outputs, split="valid") -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        # we divide by log(2) to convert the loss from base e to base 2
        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        if "valid" in split or "test" in split:
            predicts = torch.cat([log.get("predict") for log in logging_outputs], dim=0)
            if predicts.size(-1) == 1:
                # single label regression task, add aggregate acc and loss score
                targets = torch.cat(
                    [log.get("target", 0) for log in logging_outputs], dim=0
                )
                smi_list = [
                    item for log in logging_outputs for item in log.get("smi_name")
                ]
                df = pd.DataFrame(
                    {
                        "predict": predicts.view(-1).cpu(),
                        "target": targets.view(-1).cpu(),
                        "smi": smi_list,
                    }
                )
                mae = np.abs(df["predict"] - df["target"]).mean()
                mse = ((df["predict"] - df["target"]) ** 2).mean()
                metrics.log_scalar(f"{split}_mae", mae, sample_size, round=3)
                metrics.log_scalar(f"{split}_mse", mse, sample_size, round=3)
                metrics.log_scalar(
                    f"{split}_rmse", np.sqrt(mse), sample_size, round=4
                )

    @staticmethod
    def logging_outputs_can_be_summed(is_train) -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return is_train

@register_loss("finetune_mse_qm")
class FinetuneWithQMMSELoss(UnicoreLoss):
    def __init__(self, task):
        super().__init__(task)
        # self.criterion = nn.CrossEntropyLoss()

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output_a, batch_a, mulliken_pred_a, wiberg_pred_a, mulliken_charges_a, wiberg_bonds_a = model(
            sample["net_input_a"],
            classification_head_name=self.args.classification_head_name,
            features_only = True,
        )
        net_output_b, batch_b, mulliken_pred_b, wiberg_pred_b, mulliken_charges_b, wiberg_bonds_b = model(
            sample["net_input_b"],
            classification_head_name=self.args.classification_head_name,
            features_only = True,
        )

        """merge a,b"""
        mulliken_pred = pad_and_concat_2d(mulliken_pred_a,mulliken_pred_b)
        wiberg_pred = pad_and_concat_3d(wiberg_pred_a,wiberg_pred_b)
        mulliken_charges = pad_and_concat_2d(mulliken_charges_a,mulliken_charges_b)
        wiberg_bonds = pad_and_concat_3d(wiberg_bonds_a,wiberg_bonds_b)

        loss, predict = self.compute_loss(model, net_output_a, net_output_b, batch_a, batch_b, sample, reduce=reduce,)
        sample_size = sample["target"]["finetune_target"].size(0)
        if not self.training:
            if self.task.mean and self.task.std:
                targets_mean = torch.tensor(self.task.mean, device=predict.device)
                targets_std = torch.tensor(self.task.std, device=predict.device)
                predict = predict * targets_std + targets_mean
            logging_output = {
                "pka_loss": loss.data,
                "predict": predict.view(-1, self.args.num_classes).data,
                "target": sample["target"]["finetune_target"]
                .view(-1, self.args.num_classes)
                .data,
                "smi_name": sample["id"],
                "sample_size": sample_size,
                "num_task": self.args.num_classes,
                "conf_size": self.args.conf_size,
                "bsz": sample["target"]["finetune_target"].size(0),
            }
        else:
            logging_output = {
                "pka_loss": loss.data,
                "sample_size": sample_size,
                "bsz": sample["target"]["finetune_target"].size(0),
            }
        loss, logging_output = self.compute_qm_loss(loss, mulliken_pred, wiberg_pred, mulliken_charges, wiberg_bonds, logging_output, 
                                                    sample, reduce=reduce)
        logging_output['loss'] = loss.data

        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output_a, net_output_b, batch_a, batch_b, sample, reduce=True,):
        free_energy_a = net_output_a.view(-1, self.args.num_classes).float()
        free_energy_b = net_output_b.view(-1, self.args.num_classes).float()
        if not self.training:
            def compute_agg_free_energy(free_energy, batch):
                split_tensor_list = torch.split(free_energy, self.args.conf_size, dim=0)
                mean_tensor_list = [torch.mean(x, dim=0, keepdim=True) for x in split_tensor_list]
                agg_free_energy = torch.cat(mean_tensor_list, dim=0)
                agg_batch = [x//self.args.conf_size for x in batch]
                return agg_free_energy, agg_batch
            free_energy_a, batch_a = compute_agg_free_energy(free_energy_a, batch_a)
            free_energy_b, batch_b = compute_agg_free_energy(free_energy_b, batch_b)

        free_energy_a_padded = torch.nn.utils.rnn.pad_sequence(
            torch.split(free_energy_a, batch_a),
            padding_value=float("inf")
        )
        free_energy_b_padded = torch.nn.utils.rnn.pad_sequence(
            torch.split(free_energy_b, batch_b),
            padding_value=float("inf")
        )
        predicts = (
            torch.logsumexp(-free_energy_a_padded, dim=0)-
            torch.logsumexp(-free_energy_b_padded, dim=0)
        ) /  torch.log(torch.tensor([10.0])).item()

        targets = (
            sample["target"]["finetune_target"].view(-1, self.args.num_classes).float()
        )
        if self.task.mean and self.task.std:
            targets_mean = torch.tensor(self.task.mean, device=targets.device)
            targets_std = torch.tensor(self.task.std, device=targets.device)
            targets = (targets - targets_mean) / targets_std

        loss = F.mse_loss(
            predicts,
            targets,
            reduction="sum" if reduce else "none",
        )

        return loss, predicts
    

    def compute_qm_loss(self, loss, mulliken_pred, wiberg_pred, mulliken_charges, wiberg_bonds, logging_output, sample, reduce=True):

        if self.args.mulliken_charge_loss > 0:
            loss_mullikenCharges = F.mse_loss(
                mulliken_pred,
                mulliken_charges,
                reduction="none",)
            mask = (mulliken_charges != 0).float()

            loss_mullikenCharges = torch.mean(torch.sum(loss_mullikenCharges*mask, dim=-1))
            if 'mulliken_charge_loss' in logging_output:
                logging_output['mulliken_charge_loss'] += loss_mullikenCharges.data
            else:
                logging_output['mulliken_charge_loss'] = loss_mullikenCharges.data
            loss += (loss_mullikenCharges * self.args.mulliken_charge_loss)

        if self.args.wiberg_bonds_loss > 0:
            def masked_cross_entropy(y_pred, y_true, wiberg_thredhold):
                b,n,_ = y_true.shape
                if wiberg_thredhold >= 1:
                    """分bin多分类"""
                    # y_pred = y_pred.reshape(b, n*n, 3)
                    # y_true = y_true.reshape(b, n*n)

                    mask = (y_true != -100).float().view(-1)
                    loss = F.cross_entropy(y_pred.view(-1,3), y_true.view(-1).to(torch.int64), ignore_index=-100, reduction='none')
                else:
                    y_pred = y_pred.reshape(b, n*n)
                    y_true = y_true.reshape(b, n*n)
            
                    mask = (y_true != -100).float()
                    """二分类"""
                    loss = F.binary_cross_entropy_with_logits(y_pred, y_true, reduction='none')
                
                loss = loss * mask
                return loss.sum() / mask.sum()
            
            loss_wibergBonds = masked_cross_entropy(wiberg_pred, wiberg_bonds, self.args.wiberg_thredhold)
            # #  mse
            # b,n,_ = wiberg_pred.shape
            # loss_wibergBonds = F.mse_loss(
            #     wiberg_pred.reshape(b, n*n),
            #     wiberg_bonds.reshape(b, n*n),
            #     reduction="none",
            # )
            # mask = (loss_wibergBonds != 0).float()
            # loss_wibergBonds = torch.mean(torch.mean(loss_wibergBonds*mask, dim=-1))
            # loss_wibergBonds = self.wiberg_ranking_loss(wiberg_pred, wiberg_bonds, sample)
            
            if 'wiberg_bonds_loss' in logging_output:
                logging_output['wiberg_bonds_loss'] += loss_wibergBonds.data
            else:
                logging_output['wiberg_bonds_loss'] = loss_wibergBonds.data
            loss += (loss_wibergBonds * self.args.wiberg_bonds_loss)
        
        return loss, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs, split="valid") -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        # we divide by log(2) to convert the loss from base e to base 2
        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        pka_loss = sum(log.get("pka_loss", 0) for log in logging_outputs)
        metrics.log_scalar(
            "pka_loss", pka_loss / sample_size, sample_size, round=3
        )
        mulliken_charge_loss = sum(log.get("mulliken_charge_loss", 0) for log in logging_outputs)
        metrics.log_scalar(
                "mulliken_charge_loss", mulliken_charge_loss / sample_size, sample_size, round=3
        )
        wiberg_bonds_loss = sum(log.get("wiberg_bonds_loss", 0) for log in logging_outputs)
        metrics.log_scalar(
                "wiberg_bonds_loss", wiberg_bonds_loss / sample_size, sample_size, round=3
        )

        if "valid" in split or "test" in split:
            predicts = torch.cat([log.get("predict") for log in logging_outputs], dim=0)
            if predicts.size(-1) == 1:
                # single label regression task, add aggregate acc and loss score
                targets = torch.cat(
                    [log.get("target", 0) for log in logging_outputs], dim=0
                )
                smi_list = [
                    item for log in logging_outputs for item in log.get("smi_name")
                ]
                df = pd.DataFrame(
                    {
                        "predict": predicts.view(-1).cpu(),
                        "target": targets.view(-1).cpu(),
                        "smi": smi_list,
                    }
                )
                mae = np.abs(df["predict"] - df["target"]).mean()
                mse = ((df["predict"] - df["target"]) ** 2).mean()
                metrics.log_scalar(f"{split}_mae", mae, sample_size, round=3)
                metrics.log_scalar(f"{split}_mse", mse, sample_size, round=3)
                metrics.log_scalar(
                    f"{split}_rmse", np.sqrt(mse), sample_size, round=4
                )

    @staticmethod
    def logging_outputs_can_be_summed(is_train) -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return is_train

    def wiberg_ranking_loss(self, preds, targets, sample):
        atom_num_a = sample["net_input_a"][5]
        atom_num_b = sample["net_input_b"][5]
        atom_nums = torch.cat((atom_num_a, atom_num_b))
        preds_list, targets_list = self.mat2list(preds, targets, atom_nums)
        b,n,_ = targets.shape
        # wiberg_loss = self.listMLE(preds.reshape(b,n*n), targets.reshape(b,n*n))
        wiberg_loss = self.listMLE(preds_list, targets_list)
        return wiberg_loss

    def mat2list(self, preds, targets, atom_nums):
        out_preds, out_targets = [], []
        for pred, target, atom_num in zip(preds.unbind(0), targets.unbind(0), atom_nums):
            pred = pred[:atom_num, :atom_num].flatten()
            target = target[:atom_num, :atom_num].flatten()
            out_preds.append(pred)
            out_targets.append(target)
        return out_preds, out_targets


    # def listMLE(self, y_pred, y_true, eps=1e-8, padded_value_indicator=0, threshold = 0.3):
    #     """
    #     ListMLE loss introduced in "Listwise Approach to Learning to Rank - Theory and Algorithm".
    #     :param y_pred: predictions from the model, shape [batch_size, slate_length]
    #     :param y_true: ground truth labels, shape [batch_size, slate_length]
    #     :param eps: epsilon value, used for numerical stability
    #     :param padded_value_indicator: an indicator of the y_true index containing a padded item, e.g. -1
    #     :return: loss value, a torch.Tensor
    #     """
    #     # shuffle for randomised tie resolution
    #     random_indices = torch.randperm(y_pred.shape[-1])
    #     # threshold = int(threshold * y_pred.shape[-1])
    #     y_pred_shuffled = y_pred[:, random_indices]
    #     y_true_shuffled = y_true[:, random_indices]

    #     y_true_sorted, indices = y_true_shuffled.sort(descending=True, dim=-1)

    #     mask = y_true_sorted == padded_value_indicator

    #     y_pred_shuffled = -y_pred_shuffled

    #     preds_sorted_by_true = torch.gather(y_pred_shuffled, dim=1, index=indices)
    #     preds_sorted_by_true[mask] = float("-inf")

    #     max_pred_values, _ = preds_sorted_by_true.max(dim=1, keepdim=True)

    #     preds_sorted_by_true_minus_max = preds_sorted_by_true - max_pred_values

    #     cumsums = torch.cumsum(preds_sorted_by_true_minus_max.exp().flip(dims=[1]), dim=1).flip(dims=[1])

    #     observation_loss = torch.log(cumsums + eps) - preds_sorted_by_true_minus_max

    #     observation_loss[mask] = 0.0

    #     return torch.mean(torch.mean(observation_loss, dim=1))

    def listMLE(self, y_preds, y_trues, eps=1e-8, padded_value_indicator=0):
        """
        ListMLE loss introduced in "Listwise Approach to Learning to Rank - Theory and Algorithm".
        :param y_pred: predictions from the model, shape [batch_size, slate_length]
        :param y_true: ground truth labels, shape [batch_size, slate_length]
        :param eps: epsilon value, used for numerical stability
        :param padded_value_indicator: an indicator of the y_true index containing a padded item, e.g. -1
        :return: loss value, a torch.Tensor
        """
        # shuffle for randomised tie resolution
        observation_loss = []
        for y_pred, y_true in zip(y_preds, y_trues):
            if torch.all(y_true == 0): continue

            random_indices = torch.randperm(y_pred.shape[-1])
            y_pred_shuffled = y_pred.unsqueeze(0)[:, random_indices]
            y_true_shuffled = y_true.unsqueeze(0)[:, random_indices]

            y_true_sorted, indices = y_true_shuffled.sort(descending=True, dim=-1)

            preds_sorted_by_true = torch.gather(y_pred_shuffled, dim=1, index=indices)

            max_pred_values, _ = preds_sorted_by_true.max(dim=1, keepdim=True)

            preds_sorted_by_true_minus_max = preds_sorted_by_true - max_pred_values

            cumsums = torch.cumsum(preds_sorted_by_true_minus_max.exp().flip(dims=[1]), dim=1).flip(dims=[1])

            observation_loss.append(torch.mean(torch.log(cumsums + eps) - preds_sorted_by_true_minus_max))

        if len(observation_loss) == 0: return torch.tensor(0.0)

        return torch.sum(torch.stack(observation_loss))

@register_loss("finetune_confidence_bce")
class FinetuneConfidenceBCELoss(UnicoreLoss):
    def __init__(self, task):
        super().__init__(task)

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output_a, batch_a = model(
            sample["net_input_a"],
            classification_head_name=self.args.classification_head_name,
            features_only = True,
        )
        net_output_b, batch_b = model(
            sample["net_input_b"],
            classification_head_name=self.args.classification_head_name,
            features_only = True,
        )

        loss, confidence_predict, predict = self.compute_loss(model, net_output_a, net_output_b, batch_a, batch_b, sample, reduce=reduce)
        sample_size = sample["target"]["finetune_target"].size(0)
        if self.task.mean and self.task.std:
            targets_mean = torch.tensor(self.task.mean, device=predict.device)
            targets_std = torch.tensor(self.task.std, device=predict.device)
            predict = predict * targets_std + targets_mean

        if not self.training:
            logging_output = {
                "loss": loss.data,
                "predict": predict.view(-1, self.args.num_classes).data,
                "target": sample["target"]["finetune_target"]
                .view(-1, self.args.num_classes)
                .data,
                "confidence_predict": confidence_predict.view(-1, self.args.num_classes).data,
                "smi_name": sample["id"],
                "sample_size": sample_size,
                "num_task": self.args.num_classes,
                "conf_size": self.args.conf_size,
                "bsz": sample["target"]["finetune_target"].size(0),
            }
        else:
            logging_output = {
                "loss": loss.data,
                "sample_size": sample_size,
                "bsz": sample["target"]["finetune_target"].size(0),
            }
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output_a, net_output_b, batch_a, batch_b, sample, reduce=True):
        outputs_a = net_output_a.float()
        outputs_b = net_output_b.float()

        free_energy_a = outputs_a[:,0].unsqueeze(-1)
        confidence_a = outputs_a[:,1].unsqueeze(-1)
        free_energy_b = outputs_b[:,0].unsqueeze(-1)
        confidence_b = outputs_b[:,1].unsqueeze(-1)

        if not self.training:
            def compute_agg_free_energy(free_energy, batch):
                split_tensor_list = torch.split(free_energy, self.args.conf_size, dim=0)
                mean_tensor_list = [torch.mean(x, dim=0, keepdim=True) for x in split_tensor_list]
                agg_free_energy = torch.cat(mean_tensor_list, dim=0)
                agg_batch = [x//self.args.conf_size for x in batch]
                return agg_free_energy, agg_batch
            free_energy_a, _ = compute_agg_free_energy(free_energy_a, batch_a)
            free_energy_b, _ = compute_agg_free_energy(free_energy_b, batch_b)
            confidence_a, batch_a = compute_agg_free_energy(confidence_a, batch_a)
            confidence_b, batch_b = compute_agg_free_energy(confidence_b, batch_b)

        free_energy_a_padded = torch.nn.utils.rnn.pad_sequence(
            torch.split(free_energy_a, batch_a),
            padding_value=float("inf")
        )
        free_energy_b_padded = torch.nn.utils.rnn.pad_sequence(
            torch.split(free_energy_b, batch_b),
            padding_value=float("inf")
        )
        confidence_a_padded = torch.nn.utils.rnn.pad_sequence(
            torch.split(confidence_a, batch_a),
            padding_value=0.0
        )
        confidence_b_padded = torch.nn.utils.rnn.pad_sequence(
            torch.split(confidence_b, batch_b),
            padding_value=0.0
        )

        confidence_a_mask = confidence_a_padded != 0
        confidence_a_mean = torch.sum(confidence_a_padded * confidence_a_mask, dim=0) / torch.sum(confidence_a_mask, dim=0)
        confidence_b_mask = confidence_b_padded != 0
        confidence_b_mean = torch.sum(confidence_b_padded * confidence_b_mask, dim=0) / torch.sum(confidence_b_mask, dim=0)
        confidence_a_weight = torch.sum(confidence_a_mask, dim=0) / (torch.sum(confidence_a_mask, dim=0) + torch.sum(confidence_b_mask, dim=0))
        confidence_b_weight = torch.sum(confidence_b_mask, dim=0) / (torch.sum(confidence_a_mask, dim=0) + torch.sum(confidence_b_mask, dim=0))
        confidence_predict = confidence_a_mean * confidence_a_weight + confidence_b_mean * confidence_b_weight

        predicts = (
            torch.logsumexp(-free_energy_a_padded, dim=0)-
            torch.logsumexp(-free_energy_b_padded, dim=0)
        ) /  torch.log(torch.tensor([10.0])).item()

        targets = (
            sample["target"]["finetune_target"].view(-1, self.args.num_classes).float()
        )
        # if self.task.mean and self.task.std:
        #     targets_mean = torch.tensor(self.task.mean, device=targets.device)
        #     targets_std = torch.tensor(self.task.std, device=targets.device)
        #     targets = (targets - targets_mean) / targets_std

        # loss_pka = (predicts - targets)
        # loss_pka = torch.abs(loss_pka)
        # # loss_pka = torch.sqrt(loss_pka)

        # loss_confidence = torch.sqrt(loss_confidence)
        # delta = 1.0   
        # loss_confidence = F.huber_loss(confidence_predict.unsqueeze(-1), loss_pka, delta=delta)
        """
        
        """
        # thredhold = self.args.thredhold
        # binary_targets = (loss_pka <= thredhold).float()
        # print(f'positive: {sum(binary_targets)}, total:{len(binary_targets)}')
        # print(targets)
        loss_confidence = F.binary_cross_entropy_with_logits(
            confidence_predict, 
            targets,
            reduction="sum" if reduce else "none",
        )
        # loss_confidence = F.l1_loss(confidence_predict.unsqueeze(-1), loss_pka)

        return loss_confidence, confidence_predict, predicts

    @staticmethod
    def reduce_metrics(logging_outputs, split="valid") -> None:
        """Aggregate logging outputs from data parallel training."""
        def calculate_metrics(predictions, targets):
            """
            Args:
                df: Pandas DataFrame，包含预测值和真实值的列
                pred_col: str，预测值列名
                target_col: str，真实值列名
            Returns:
                metrics: dict，包含 Accuracy, Recall, F1 Score
            """
            predictions = (predictions > 0.5)
            accuracy = accuracy_score(targets, predictions)
            recall = recall_score(targets, predictions)
            f1 = f1_score(targets, predictions)
            
            return accuracy, recall, f1

        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        # we divide by log(2) to convert the loss from base e to base 2
        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        if "valid" in split or "test" in split:
            predicts = torch.cat([log.get("confidence_predict") for log in logging_outputs], dim=0)
            if predicts.size(-1) == 1:
                # single label regression task, add aggregate acc and loss score
                targets = torch.cat(
                    [log.get("target", 0) for log in logging_outputs], dim=0
                )
                smi_list = [
                    item for log in logging_outputs for item in log.get("smi_name")
                ]
                df = pd.DataFrame(
                    {
                        "predict": torch.sigmoid(predicts).view(-1).cpu(),
                        "target": targets.view(-1).cpu(),
                        "smi": smi_list,
                    }
                )
                # mae = np.abs(df["predict"] - df["target"]).mean()
                # mse = ((df["predict"] - df["target"]) ** 2).mean()
                print(f"pred positive: {sum(df['predict'].values > 0.5)}")
                print(f"target positive: {sum(df['target'].values)}")
                acc, recall, f1 = calculate_metrics(df["predict"].values, df["target"].values)
                print(f"acc: {acc}, recall: {recall}, f1: {f1}")
                metrics.log_scalar(f"{split}_mae", acc, sample_size, round=3)
                metrics.log_scalar(f"{split}_mse", recall, sample_size, round=3)
                metrics.log_scalar(
                    f"{split}_rmse", f1, sample_size, round=4
                )

    @staticmethod
    def logging_outputs_can_be_summed(is_train) -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return is_train

@register_loss("finetune_logd_mse")
class FinetunelogDMSELoss(UnicoreLoss):
    def __init__(self, task):
        super().__init__(task)
    
    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output, batch = model(
            sample["net_input"],
            classification_head_name=self.args.classification_head_name,
            features_only = True,
        )

        loss, predict, loss_d, loss_p = self.compute_loss(model, net_output, batch, sample, reduce=reduce)
        sample_size = sample["target"]["finetune_target"].size(0)
        if not self.training:
            if self.task.logd_mean and self.task.logd_std:
                logd_targets_mean = torch.tensor(self.task.logd_mean, device=predict.device)
                logd_targets_std = torch.tensor(self.task.logd_std, device=predict.device)
                predict[:,0] = (predict[:,0] * logd_targets_std) + logd_targets_mean
                
            if self.task.logp_mean and self.task.logp_std:
                logp_targets_mean = torch.tensor(self.task.logp_mean, device=predict.device)
                logp_targets_std = torch.tensor(self.task.logp_std, device=predict.device)   
                predict[:,1] = (predict[:,1] * logp_targets_std) + logp_targets_mean

            logging_output = {
                "loss": loss.data,
                "loss_d": loss_d.data,
                "loss_p": loss_p.data,
                "predict": predict.view(-1, self.args.num_classes).data,
                "target": sample["target"]["finetune_target"]
                .view(-1, self.args.num_classes)
                .data,
                "smi_name": sample["id"],
                "sample_size": sample_size,
                "num_task": self.args.num_classes,
                "conf_size": self.args.conf_size,
                "bsz": sample["target"]["finetune_target"].size(0),
                "logD": self.args.logD,
                "logP": self.args.logP,
            }
        else:
            logging_output = {
                "loss": loss.data,
                "sample_size": sample_size,
                "bsz": sample["target"]["finetune_target"].size(0),
            }
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, batch, sample, reduce=True):
        logd_logp = net_output.view(-1, self.args.num_classes).float()
        if not self.training:
            def compute_agg_free_energy(free_energy, batch):
                split_tensor_list = torch.split(free_energy, self.args.conf_size, dim=0)
                mean_tensor_list = [torch.mean(x, dim=0, keepdim=True) for x in split_tensor_list]
                agg_free_energy = torch.cat(mean_tensor_list, dim=0)
                agg_batch = [x//self.args.conf_size for x in batch]
                return agg_free_energy, agg_batch
            logd_logp, batch = compute_agg_free_energy(logd_logp, batch)
        
        targets = (
            sample["target"]["finetune_target"].view(-1, self.args.num_classes).float()
        )
        logd_targets = targets[:,0]
        logd_preds = logd_logp[:,0]
        mask_d = logd_targets != 0
        
        logp_targets = targets[:,1]
        logp_preds = logd_logp[:,1]
        mask_p = logp_targets != 0

        if self.task.logd_mean and self.task.logd_std:
            logd_targets_mean = torch.tensor(self.task.logd_mean, device=targets.device)
            logd_targets_std = torch.tensor(self.task.logd_std, device=targets.device)
            logd_targets = (logd_targets - logd_targets_mean) / logd_targets_std

        if self.task.logp_mean and self.task.logp_std:
            logp_targets_mean = torch.tensor(self.task.logp_mean, device=targets.device)
            logp_targets_std = torch.tensor(self.task.logp_std, device=targets.device)
            logp_targets = (logp_targets - logp_targets_mean) / logp_targets_std

        loss_logd = torch.tensor(0.0, requires_grad=True)
        if self.args.logD and sum(mask_d) !=0:
            loss_logd = F.mse_loss(
                logd_preds[mask_d],
                logd_targets[mask_d],
                reduction="sum" if reduce else "none",
            )
        
        loss_logp = torch.tensor(0.0, requires_grad=True)
        if self.args.logP and sum(mask_p) != 0:
            loss_logp = F.mse_loss(
                logp_preds[mask_p],
                logp_targets[mask_p],
                reduction="sum" if reduce else "none",
            )
        loss = self.args.logd_weight * loss_logd + (1-self.args.logd_weight) * loss_logp
        # loss = loss_logd + loss_logp
        return loss, logd_logp, loss_logd, loss_logp

    @staticmethod
    def reduce_metrics(logging_outputs, split="valid") -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        # we divide by log(2) to convert the loss from base e to base 2
        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        if "valid" in split or "test" in split:
            predicts = torch.cat([log.get("predict") for log in logging_outputs], dim=0)
            if predicts.size(-1) == 2:
                # single label regression task, add aggregate acc and loss score
                targets = torch.cat(
                    [log.get("target", 0) for log in logging_outputs], dim=0
                )
                smi_list = [
                    item for log in logging_outputs for item in log.get("smi_name")
                ]
                df = pd.DataFrame(
                    {
                        "logd_predict": predicts[:,0].cpu(),
                        "logd_target": targets[:,0].cpu(),
                        "logp_predict": predicts[:,1].cpu(),
                        "logp_target": targets[:,1].cpu(),
                        "smi": smi_list,
                    }
                )
                logd_mask = df["logd_target"] != 0
                logp_mask = df["logp_target"] !=0

                flag = 0
                if logging_outputs[0].get("logD", 0) and sum(logd_mask) != 0:
                    logd_mae = np.abs(df["logd_predict"][logd_mask]- df["logd_target"][logd_mask]).mean()
                    logd_mse = ((df["logd_predict"][logd_mask] - df["logd_target"][logd_mask]) ** 2).mean()
                    flag+=1
                else:
                    logd_mae = 0.0
                    logd_mse = 0.0 

                if logging_outputs[0].get("logP", 0) and sum(logp_mask) != 0:
                    logp_mae = np.abs(df["logp_predict"][logp_mask]- df["logp_target"][logp_mask]).mean()
                    logp_mse = ((df["logp_predict"][logp_mask] - df["logp_target"][logp_mask]) ** 2).mean()
                    flag+=1
                else:
                    logp_mae = 0.0
                    logp_mse = 0.0
                
                mae = (logd_mae + logp_mae) / flag
                mse = (logd_mse + logp_mse) / flag

                metrics.log_scalar(f"{split}_mae", mae, sample_size, round=3)
                metrics.log_scalar(f"{split}_mse", mse, sample_size, round=3)
                metrics.log_scalar(
                    f"{split}_rmse", np.sqrt(mse), sample_size, round=4
                )

    @staticmethod
    def logging_outputs_can_be_summed(is_train) -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return is_train

@register_loss("infer_pka_mae")
class InferPkaMaeLoss(UnicoreLoss):
    """预测pka，并还原为IMDB格式数据"""
    def __init__(self, task):
        super().__init__(task)

    def forward(self, model, sample, reduce=True):

        net_output_a, batch_a = model(
            sample["net_input_a"],
            classification_head_name=self.args.classification_head_name,
            features_only = True,
        )
        net_output_b, batch_b = model(
            sample["net_input_b"],
            classification_head_name=self.args.classification_head_name,
            features_only = True,
        )

        loss, predict = self.compute_loss(model, net_output_a, net_output_b, batch_a, batch_b, sample, reduce=reduce)
        sample_size = sample["target"]["finetune_target"].size(0)
        if not self.training:
            if self.task.mean and self.task.std:
                targets_mean = torch.tensor(self.task.mean, device=predict.device)
                targets_std = torch.tensor(self.task.std, device=predict.device)
                predict = predict * targets_std + targets_mean

            mae = np.abs(predict.view(-1, self.args.num_classes).data.cpu().numpy() - sample["target"]["finetune_target"].view(-1, self.args.num_classes).data.cpu().numpy())
            logging_output = {
                "loss": loss.data,
                "predict": predict.view(-1, self.args.num_classes).data,
                "target": sample["target"]["finetune_target"]
                .view(-1, self.args.num_classes)
                .data,
                "mae": mae,
                "smi_name": sample["id"],
                "sample_size": sample_size,
                "num_task": self.args.num_classes,
                "conf_size": self.args.conf_size,
                "bsz": sample["target"]["finetune_target"].size(0),
            }
        else:
            logging_output = {
                "loss": loss.data,
                "sample_size": sample_size,
                "bsz": sample["target"]["finetune_target"].size(0),
            }

        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output_a, net_output_b, batch_a, batch_b, sample, reduce=True):
        free_energy_a = net_output_a.view(-1, self.args.num_classes).float()
        free_energy_b = net_output_b.view(-1, self.args.num_classes).float()
        if not self.training:
            def compute_agg_free_energy(free_energy, batch):
                split_tensor_list = torch.split(free_energy, self.args.conf_size, dim=0)
                mean_tensor_list = [torch.mean(x, dim=0, keepdim=True) for x in split_tensor_list]
                agg_free_energy = torch.cat(mean_tensor_list, dim=0)
                agg_batch = [x//self.args.conf_size for x in batch]
                return agg_free_energy, agg_batch
            free_energy_a, batch_a = compute_agg_free_energy(free_energy_a, batch_a)
            free_energy_b, batch_b = compute_agg_free_energy(free_energy_b, batch_b)

        free_energy_a_padded = torch.nn.utils.rnn.pad_sequence(
            torch.split(free_energy_a, batch_a),
            padding_value=float("inf")
        )
        free_energy_b_padded = torch.nn.utils.rnn.pad_sequence(
            torch.split(free_energy_b, batch_b),
            padding_value=float("inf")
        )
        predicts = (
            torch.logsumexp(-free_energy_a_padded, dim=0)-
            torch.logsumexp(-free_energy_b_padded, dim=0)
        ) /  torch.log(torch.tensor([10.0])).item()

        targets = (
            sample["target"]["finetune_target"].view(-1, self.args.num_classes).float()
        )
        if self.task.mean and self.task.std:
            targets_mean = torch.tensor(self.task.mean, device=targets.device)
            targets_std = torch.tensor(self.task.std, device=targets.device)
            targets = (targets - targets_mean) / targets_std
        loss = F.mse_loss(
            predicts,
            targets,
            reduction="sum" if reduce else "none",
        )
        return loss, predicts

    @staticmethod
    def reduce_metrics(logging_outputs, split="valid") -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        # we divide by log(2) to convert the loss from base e to base 2
        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        if "valid" in split or "test" in split:
            predicts = torch.cat([log.get("predict") for log in logging_outputs], dim=0)
            if predicts.size(-1) == 1:
                # single label regression task, add aggregate acc and loss score
                targets = torch.cat(
                    [log.get("target", 0) for log in logging_outputs], dim=0
                )
                smi_list = [
                    item for log in logging_outputs for item in log.get("smi_name")
                ]
                df = pd.DataFrame(
                    {
                        "predict": predicts.view(-1).cpu(),
                        "target": targets.view(-1).cpu(),
                        "smi": smi_list,
                    }
                )
                mae = np.abs(df["predict"] - df["target"]).mean()
                mse = ((df["predict"] - df["target"]) ** 2).mean()
                metrics.log_scalar(f"{split}_mae", mae, sample_size, round=3)
                metrics.log_scalar(f"{split}_mse", mse, sample_size, round=3)
                metrics.log_scalar(
                    f"{split}_rmse", np.sqrt(mse), sample_size, round=4
                )

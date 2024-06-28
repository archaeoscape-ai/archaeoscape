from typing import Any, Dict, List, Tuple

import hydra
import numpy as np
import pandas as pd
import sklearn.metrics
import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification import JaccardIndex
from torchmetrics.classification.accuracy import Accuracy

from src.utils import mkdir, pylogger
from src.utils.data import Window_writer_rasterio
from src.utils.geoaffine import Interpolation, reproject_to_tcell_rasterize

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


def write_to_rasterio(wwr, preds_np, metas, poly_aoi, overlap=0.0):
    """Writes the predicted scores to a raster using the rasterio library.

    Args:
        wwr (rasterio.io.DatasetWriter): The rasterio dataset writer object.
        preds_np (numpy.ndarray): The predicted scores.
        metas (list): List of metadata dictionaries.
        poly_aoi (shapely.geometry.Polygon): The area of interest polygon.
        overlap (float, optional): The overlap to use when writing to the raster. Defaults to 0.0.

    Returns:
        None
    """
    for scores, meta in zip(preds_np, metas):
        window_icell = meta["window"]
        safft_icell_to_tcell = meta["safft_icell_to_tcell"]
        safft_world_to_icell = meta["safft_world_to_icell"]
        square = meta["square"]
        if overlap > 0.0:
            # extract the center part of the square (square is an shapely polygon)
            square_side = square.area**0.5
            square = square.buffer(-overlap * square_side / 2.0)

        goodpoly_world = poly_aoi & square
        if goodpoly_world.is_empty:
            continue
        scores_icell = np.ma.stack(
            [
                reproject_to_tcell_rasterize(
                    score,
                    window_icell,
                    safft_icell_to_tcell,
                    goodpoly_world,
                    safft_world_to_icell,
                    Interpolation.LINEAR,
                )
                for score in scores
            ]
        )
        argmax_data_icell = np.ma.argmax(scores_icell, axis=0)
        argmax_mask_icell = np.any(scores_icell.mask, axis=0)
        wwr.write(
            np.ma.MaskedArray(data=argmax_data_icell, mask=argmax_mask_icell),
            window_icell,
        )


class SegmentationModule(LightningModule):
    """Example of a `LightningModule` for MNIST classification."""

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        compile: bool,
        scheduler: torch.optim.lr_scheduler = None,
        warmup_scheduler: torch.optim.lr_scheduler = None,
        metric_monitored: str = None,
        loss_ce_weight: torch.tensor = None,
        num_classes: int = 10,
        test_overlap=0.0,
        save_freq: int = 1,
        save_eval_only: bool = False,
    ) -> None:
        """Initialize a `segmentation Module`.

        Args:
            net (torch.nn.Module): The neural network model.
            optimizer (torch.optim.Optimizer): The optimizer.
            compile (bool): Whether to compile the model.
            scheduler (torch.optim.lr_scheduler, optional): The learning rate scheduler. Defaults to None.
            warmup_scheduler (torch.optim.lr_scheduler, optional): The learning rate warmup scheduler. Defaults to None.
            metric_monitored (str, optional): The metric to monitor. Defaults to None.
            loss_ce_weight (torch.tensor, optional): The class weights for the cross-entropy loss. Defaults to None.
            num_classes (int, optional): The number of classes. Defaults to 10.
            test_overlap (float, optional): The overlap to use when writing the predictions to a raster. Defaults to 0.0.
            save_freq (int, optional): The frequency to save the predictions. Defaults to 1.
            save_eval_only (bool, optional): Whether to only save the predictions for the evaluation set. Defaults to False.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=["net"])

        self.net = net

        self.save_freq = save_freq
        self.save_val_only = save_eval_only

        # loss function
        self.criterion = torch.nn.CrossEntropyLoss(
            loss_ce_weight, reduction="mean", ignore_index=255
        )

        # metric objects for calculating and averaging accuracy across batches
        self.train_acc = Accuracy(
            task="multiclass", num_classes=num_classes, ignore_index=255
        )
        self.eval_train_acc = Accuracy(
            task="multiclass", num_classes=num_classes, ignore_index=255
        )
        self.val_acc = Accuracy(
            task="multiclass", num_classes=num_classes, ignore_index=255
        )
        # self.test_acc = Accuracy(task="multiclass", num_classes=num_classes)

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.eval_train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        # self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_acc_best = MaxMetric()

        # for per aoi evaluation
        self.last_aoi_name = None
        self.val_JaccardIndex = torch.nn.ModuleDict(
            {
                "Total": JaccardIndex(
                    task="multiclass",
                    num_classes=num_classes,
                    average="none",
                    ignore_index=255,
                )
            }
        )
        self.train_JaccardIndex = torch.nn.ModuleDict(
            {
                "Total": JaccardIndex(
                    task="multiclass",
                    num_classes=num_classes,
                    average="none",
                    ignore_index=255,
                )
            }
        )

        try:
            self.output_dir = (
                hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
            )
        except AttributeError:
            self.output_dir = "./logs"
        mkdir(self.output_dir + "/preds")

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net, backend="inductor")

    def forward(self, x: torch.Tensor, metas: List = None) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :return: A tensor of logits.
        """
        return self.net(x, metas=metas)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_acc.reset()
        self.val_acc_best.reset()
        for metric in self.val_JaccardIndex.values():
            metric.reset()
        for metric in self.train_JaccardIndex.values():
            metric.reset()

        self.net.train()

    def model_step(self, batch, overlap=-1.0):
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        data, targets, metas = batch
        output = self.forward(data, metas=metas)
        aux_loss = 0

        logics = output["out"]
        if overlap > 0.0:
            # only keep the center of the image
            h, w = targets.shape[-2:]
            new_h, new_w = h - int(h * overlap), h - int(w * overlap)
            h_start, w_start = int(h * overlap) // 2, int(w * overlap) // 2
            targets = targets[
                ..., h_start : h_start + new_h, w_start : w_start + new_w
            ]
            logics = logics[
                ..., h_start : h_start + new_h, w_start : w_start + new_w
            ]

        loss = self.criterion(logics, targets)

        preds = torch.nn.functional.softmax(logics, dim=1)
        if overlap >= 0.0:
            preds_full = torch.nn.functional.softmax(output["out"], dim=1)
            return loss, preds, targets, metas, preds_full
        return (
            loss,
            preds,
            targets,
            metas,
        )

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        (
            loss,
            preds,
            targets,
            _,
        ) = self.model_step(batch)
        preds_int = preds.argmax(axis=1)

        # update and log metrics
        self.train_loss(loss)
        self.train_acc(preds_int, targets)
        self.log(
            "train/loss",
            self.train_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "train/acc",
            self.train_acc,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )

        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass

    def validation_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        # common per aoi
        aoi_name = batch[2][0]["aoi_name"]
        gf_aoi = self.trainer.datamodule.gf_aois.loc[aoi_name]
        poly_aoi = gf_aoi["geometry"]
        split = gf_aoi["split"]
        raster_targets = self.trainer.datamodule.raster_targets
        epoch = self.trainer.current_epoch

        # all the righting to raster is done on rank 0
        new_aoi = self.last_aoi_name != aoi_name
        correct_split = not self.save_val_only or split == "val"
        correct_epoch = (self.trainer.max_epochs - epoch) % self.save_freq == 0
        rank_zero = self.trainer.global_rank == 0
        multi_gpu = self.trainer.world_size > 1

        if new_aoi and correct_split and correct_epoch and rank_zero:
            raster_argmax_path = (
                self.output_dir
                + f"/preds/eval_{self.trainer.current_epoch}_{aoi_name}_argmax.tif"
            )
            if self.last_aoi_name is not None:
                self.wwr.close()
            profile = raster_targets[aoi_name]["profile"]
            profile["count"] = 1
            self.wwr = Window_writer_rasterio(
                raster_argmax_path, profile, "uint8"
            )
            self.last_aoi_name = aoi_name

        loss, preds, targets, metas = self.model_step(batch)
        preds_int = preds.argmax(axis=1)

        if correct_split and correct_epoch:
            if multi_gpu:
                # Gather from all workers
                preds_all = self.all_gather(preds)
                metas_all = [_ for _ in range(self.trainer.world_size)]
                torch.distributed.all_gather_object(metas_all, metas)
                preds_all = preds_all.reshape(-1, *preds_all.shape[2:])
                metas_all = [item for sublist in metas_all for item in sublist]
            else:
                preds_all = preds
                metas_all = metas

            if rank_zero:
                write_to_rasterio(
                    self.wwr,
                    preds_all.detach().cpu().numpy(),
                    metas_all,
                    poly_aoi,
                )

        # update and log metrics
        if split == "train":
            self.eval_train_loss(loss)
            self.eval_train_acc(preds_int, targets)
            # IoU
            if aoi_name not in self.train_JaccardIndex:
                self.train_JaccardIndex[aoi_name] = JaccardIndex(
                    task="multiclass",
                    num_classes=self.hparams.num_classes,
                    average="none",
                    ignore_index=255,
                ).to(self.device)
            self.train_JaccardIndex[aoi_name].update(preds_int, targets)
            self.train_JaccardIndex["Total"].update(preds_int, targets)

        elif split == "val":
            self.val_loss(loss)
            self.val_acc(preds_int, targets)
            # IoU
            if aoi_name not in self.val_JaccardIndex:
                self.val_JaccardIndex[aoi_name] = JaccardIndex(
                    task="multiclass",
                    num_classes=self.hparams.num_classes,
                    average="none",
                    ignore_index=255,
                ).to(self.device)
            self.val_JaccardIndex[aoi_name].update(preds_int, targets)
            self.val_JaccardIndex["Total"].update(preds_int, targets)

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        acc = self.val_acc.compute()  # get current val acc
        self.val_acc_best(acc)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log(
            "val/loss", self.val_loss.compute(), sync_dist=True, prog_bar=True
        )
        self.log(
            "val/acc", self.val_acc.compute(), sync_dist=True, prog_bar=True
        )
        self.log(
            "val/acc_best",
            self.val_acc_best.compute(),
            sync_dist=True,
            prog_bar=True,
        )
        # self.log(
        #     "eval_train/loss",
        #     self.eval_train_loss.compute(),
        #     sync_dist=True,
        #     prog_bar=False,
        # )
        # self.log(
        #     "eval_train/acc",
        #     self.eval_train_acc.compute(),
        #     sync_dist=True,
        #     prog_bar=True,
        # )

        # log IoU to logger
        # for aoi_name, metric in self.train_JaccardIndex.items():
        #     # self.log(f"Train IoU {aoi_name}", metric.compute(), sync_dist=True, prog_bar=False)
        # for aoi_name, metric in self.eval_JaccardIndex.items():
        #     # self.log(f"Eval IoU {aoi_name}", metric.compute(), sync_dist=True, prog_bar=False)

        # logging to info
        epoch = self.trainer.current_epoch
        log.info(f"{epoch=}")
        log.info(
            f"eval_train/loss={self.train_loss.compute()}, eval_train/acc={self.eval_train_acc.compute()} val/loss={self.val_loss.compute()}, val/acc={self.val_acc.compute()}"
        )

        # IoU
        # log.info('labels_names')
        metrics = pd.DataFrame()
        label_names = self.trainer.datamodule.label_names + ["background"]
        # set data columns names to label names
        for aoi_name, metric in self.val_JaccardIndex.items():
            metric_l = metric.compute().detach().cpu().numpy()
            row = {
                label_names[i]: metric_l[i]
                for i in range(self.hparams.num_classes)
            }
            row.update({"mean": metric_l.mean()})
            df = pd.DataFrame(row, index=[f"Eval IoU {aoi_name}"])
            metrics = pd.concat([metrics, df])
            # Log to logger
            if aoi_name == "Total":
                self.log(
                    "val/IoU/mean",
                    metric_l.mean(),
                    sync_dist=True,
                    prog_bar=True,
                )
                for i in range(self.hparams.num_classes):
                    self.log(
                        f"val/IoU/{label_names[i]}",
                        metric_l[i],
                        sync_dist=True,
                        prog_bar=False,
                    )
        for aoi_name, metric in self.train_JaccardIndex.items():
            metric_l = metric.compute().detach().cpu().numpy()
            row = {
                label_names[i]: metric_l[i]
                for i in range(self.hparams.num_classes)
            }
            row.update({"mean": metric_l.mean()})
            df = pd.DataFrame(row, index=[f"Train IoU {aoi_name}"])
            metrics = pd.concat([metrics, df])
            # Log to logger
            if aoi_name == "Total":
                self.log(
                    "eval_train/IoU/mean",
                    metric_l.mean(),
                    sync_dist=True,
                    prog_bar=False,
                )
                for i in range(self.hparams.num_classes):
                    self.log(
                        f"eval_train/IoU/{label_names[i]}",
                        metric_l[i],
                        sync_dist=True,
                        prog_bar=False,
                    )
        log.info("\n" + metrics.to_markdown())

        # reset metrics
        self.val_loss.reset()
        self.val_acc.reset()
        self.eval_train_loss.reset()
        self.eval_train_acc.reset()
        for metric in self.val_JaccardIndex.values():
            metric.reset()
        for metric in self.train_JaccardIndex.values():
            metric.reset()
        if self.last_aoi_name is not None:
            self.wwr.close()

    def test_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        # Multi gpu
        rank_zero = self.trainer.global_rank == 0
        multi_gpu = self.trainer.world_size > 1

        # common per aoi
        aoi_name = batch[2][0]["aoi_name"]
        gf_aoi = self.trainer.datamodule.gf_aois.loc[aoi_name]
        poly_aoi = gf_aoi["geometry"]
        raster_targets = self.trainer.datamodule.raster_targets

        # create dir if needed
        mkdir(self.output_dir + "/test")

        raster_argmax_path = self.output_dir + f"/test/{aoi_name}_argmax.tif"
        if self.last_aoi_name != aoi_name:
            if self.last_aoi_name is not None:
                self.wwr.close()
            profile = raster_targets[aoi_name]["profile"]
            profile["count"] = 1
            self.wwr = Window_writer_rasterio(
                raster_argmax_path, profile, "uint8"
            )
            self.last_aoi_name = aoi_name

        loss, preds, targets, metas, preds_full = self.model_step(
            batch, overlap=self.hparams.test_overlap
        )
        preds_int = preds.argmax(axis=1)

        if multi_gpu:
            # Gather from all workers
            preds_all = self.all_gather(preds_full)
            metas_all = [_ for _ in range(self.trainer.world_size)]
            torch.distributed.all_gather_object(metas_all, metas)
            preds_all = preds_all.reshape(-1, *preds_all.shape[2:])
            metas_all = [item for sublist in metas_all for item in sublist]
        else:
            preds_all = preds_full
            metas_all = metas

        if rank_zero:
            write_to_rasterio(
                self.wwr,
                preds_all.detach().cpu().numpy(),
                metas_all,
                poly_aoi,
                overlap=self.hparams.test_overlap,
            )

        # create metrics if not already created
        if not hasattr(self, "test_loss"):
            self.test_loss = MeanMetric().to(self.device)
            self.test_acc = Accuracy(
                task="multiclass",
                num_classes=self.hparams.num_classes,
                ignore_index=255,
            ).to(self.device)
            self.test_JaccardIndex = torch.nn.ModuleDict(
                {
                    "Total": JaccardIndex(
                        task="multiclass",
                        num_classes=self.hparams.num_classes,
                        average="none",
                        ignore_index=255,
                    )
                }
            ).to(self.device)

        # update and log metrics
        self.test_loss(loss)
        self.test_acc(preds_int, targets)
        # IoU
        if aoi_name not in self.test_JaccardIndex:
            self.test_JaccardIndex[aoi_name] = JaccardIndex(
                task="multiclass",
                num_classes=self.hparams.num_classes,
                average="none",
                ignore_index=255,
            ).to(self.device)
        self.test_JaccardIndex[aoi_name].update(preds_int, targets)
        self.test_JaccardIndex["Total"].update(preds_int, targets)

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        # log metrics
        self.log("test/loss", self.test_loss.compute(), sync_dist=True)
        self.log("test/acc", self.test_acc.compute(), sync_dist=True)
        # log metrics to info
        log.info(
            f"test/loss={self.test_loss.compute()}, test/acc={self.test_acc.compute()}"
        )
        # log IoU to logger
        # log.info('labels_names')
        metrics = pd.DataFrame()
        label_names = self.trainer.datamodule.label_names + ["background"]
        # set data columns names to label names
        for aoi_name, metric in self.test_JaccardIndex.items():
            metric_l = metric.compute().detach().cpu().numpy()
            row = {
                label_names[i]: metric_l[i]
                for i in range(self.hparams.num_classes)
            }
            row.update({"mean": metric_l.mean()})
            df = pd.DataFrame(row, index=[f"Eval IoU {aoi_name}"])
            metrics = pd.concat([metrics, df])
        log.info("\n" + metrics.to_markdown())
        self.wwr.close()

    def predict_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        """Perform a single predict step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        # common per aoi
        aoi_name = batch[2][0]["row_square"]["aoi_name"]
        gf_aoi = self.trainer.datamodule.gf_aois.loc[aoi_name]
        poly_aoi = gf_aoi["geometry"]
        split = gf_aoi["split"]
        raster_targets = self.trainer.datamodule.raster_targets

        raster_argmax_path = (
            self.output_dir + f"/preds/prediction_{aoi_name}_argmax.tif"
        )
        if self.last_aoi_name != aoi_name:
            if self.last_aoi_name is not None:
                self.wwr.close()
            profile = raster_targets[aoi_name]["profile"]
            profile["count"] = 1
            self.wwr = Window_writer_rasterio(
                raster_argmax_path, profile, "uint8"
            )
            self.last_aoi_name = aoi_name

        loss, preds, targets, metas = self.model_step(batch)
        preds_int = preds.argmax(axis=1)

        write_to_rasterio(
            self.wwr, preds.detach().cpu().numpy(), metas, poly_aoi
        )

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(
            params=self.trainer.model.parameters()
        )
        list_schedulers = []
        if self.hparams.scheduler is not None:
            assert (
                self.hparams.metric_monitored is not None
            ), "A metric must be monitored to use a scheduler."
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            list_schedulers.append(
                {
                    "scheduler": scheduler,
                    "monitor": self.hparams.metric_monitored,
                    "interval": "epoch",
                    "frequency": 1,
                }
            )
        if self.hparams.warmup_scheduler is not None:
            warmup = self.hparams.warmup_scheduler(
                optimizer=optimizer,
                total_step=self.trainer.estimated_stepping_batches,
            )
            list_schedulers.append(
                {
                    "scheduler": warmup,
                    "interval": "step",
                    "frequency": 1,
                }
            )
        print(f"{list_schedulers=}")
        return [optimizer], list_schedulers


if __name__ == "__main__":
    _ = SegmentationModule(None, None, None, None)

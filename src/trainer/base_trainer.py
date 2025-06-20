from abc import abstractmethod

import torch
from tqdm.auto import tqdm

from src.datasets.data_utils import inf_loop
from src.metrics.tracker import MetricTracker
from src.model.ema import EMA
from src.utils.io_utils import ROOT_PATH


class BaseTrainer:
    """
    Base class for all trainers.
    """

    def __init__(
        self,
        model,
        criterion,
        metrics,
        optimizer,
        # lr_scheduler,
        config,
        device,
        dataloaders,
        logger,
        writer,
    ):
        """
        Args:
            model (nn.Module): PyTorch model.
            criterion (nn.Module): loss function for model training.
            metrics (dict): dict with the definition of metrics for training
                (metrics[train]) and inference (metrics[inference]). Each
                metric is an instance of src.metrics.BaseMetric.
            optimizer (Optimizer): optimizer for the model.
            lr_scheduler (LRScheduler): learning rate scheduler for the
                optimizer.
            config (DictConfig): experiment config containing training config.
            device (str): device for tensors and model.
            dataloaders (dict[DataLoader]): dataloaders for different
                sets of data.
            logger (Logger): logger that logs output.
            writer (WandBWriter | CometMLWriter): experiment tracker.
            epoch_len (int | None): number of steps in each epoch for
                iteration-based training. If None, use epoch-based
                training (len(dataloader)).
            skip_oom (bool): skip batches with the OutOfMemory error.
            batch_transforms (dict[Callable] | None): transforms that
                should be applied on the whole batch. Depend on the
                tensor name.
        """
        self.is_train = True
        self.discriminator_steps = (
            config.trainer.discriminator_steps
            if config.trainer.discriminator_steps is not None
            else 1
        )

        self.config = config
        self.cfg_trainer = self.config.trainer

        self.device = device

        self.logger = logger
        self.log_step = config.trainer.get("log_step", 50)

        self.optimizer = optimizer
        # self.lr_scheduler = lr_scheduler
        # self.batch_transforms = batch_transforms

        # define dataloaders
        self.train_dataloader = dataloaders["train"]
        self.reference_dataloader = inf_loop(dataloaders["reference"])
        self.epoch_len = len(self.train_dataloader)
        self.evaluation_dataloader = dataloaders["val"]
        self.model = model
        self.criterion = criterion
        self.criterion["generator_loss"].set_steps(self.epoch_len)
        if device == "cuda":
            self.model.generator = torch.compile(self.model.generator)
            self.model.mapping_network = torch.compile(self.model.mapping_network)
            self.model.style_encoder = torch.compile(self.model.style_encoder)
            self.criterion["generator_loss"] = torch.compile(
                self.criterion["generator_loss"]
            )

        # define epochs
        self._last_epoch = 0  # required for saving on interruption
        self.start_epoch = 1
        self.epochs = self.cfg_trainer.n_epochs

        self.ema_step = config.get("ema_step", None)
        if self.ema_step is not None:
            self.gema = EMA(
                self.model.generator,
                config.initial_decay,
                config.final_decay,
                self.epochs,
            )
            self.fema = EMA(
                self.model.mapping_network,
                config.initial_decay,
                config.final_decay,
                self.epochs,
            )
            self.eema = EMA(
                self.model.style_encoder,
                config.initial_decay,
                config.final_decay,
                self.epochs,
            )

        # configuration to monitor model performance and save best

        self.save_period = (
            self.cfg_trainer.save_period
        )  # checkpoint each save_period epochs

        # setup visualization writer instance
        self.writer = writer

        # define metrics
        self.metrics = metrics
        self.train_metrics = MetricTracker(
            *self.config.writer.loss_names,
            "grad_norm",
            "g_grad_norm",
            "e_grad_norm",
            "f_grad_norm",
            "d_grad_norm",
            *[m.name for m in self.metrics["train"]],
            writer=self.writer,
        )
        self.evaluation_metrics = MetricTracker(
            *[m.name for m in self.metrics["eval"]],
            writer=self.writer,
        )

        # define checkpoint dir and init everything if required

        self.checkpoint_dir = (
            ROOT_PATH / config.trainer.save_dir / config.writer.run_name
        )

        if config.trainer.get("resume_from") is not None:
            resume_path = self.checkpoint_dir / str(config.trainer.resume_from)
            self._resume_checkpoint(resume_path)

        if config.trainer.get("from_pretrained") is not None:
            self._from_pretrained(config.trainer.get("from_pretrained"))

    def train(self):
        """
        Wrapper around training process to save model on keyboard interrupt.
        """
        try:
            self._train_process()
        except KeyboardInterrupt as e:
            self.logger.info("Saving model on keyboard interrupt")
            self._save_checkpoint(self._last_epoch, save_best=False)
            raise e

    def _train_process(self):
        """
        Full training logic:

        Training model for an epoch, evaluating it on non-train partitions,
        and monitoring the performance improvement (for early stopping
        and saving the best checkpoint).
        """
        for epoch in range(self.start_epoch, self.epochs + 1):
            self._last_epoch = epoch
            result = self._train_epoch(epoch)
            if self.ema_step:
                self.eema.update_decay()
                self.fema.update_decay()
                self.gema.update_decay()

            # save logged information into logs dict
            logs = {"epoch": epoch}
            logs.update(result)

            # print logged information to the screen
            for key, value in logs.items():
                self.logger.info(f"    {key:15s}: {value}")

            # evaluate model performance according to configured metric,
            # save best checkpoint as model_best
            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch, including logging and evaluation on
        non-train partitions.

        Args:
            epoch (int): current training epoch.
        Returns:
            logs (dict): logs that contain the average loss and metric in
                this epoch.
        """
        self.is_train = True
        self.model.train()
        self.train_metrics.reset()
        self.writer.set_step((epoch - 1) * self.epoch_len)
        self.writer.add_scalar("epoch", epoch)
        for batch_idx, batch in enumerate(
            tqdm(self.train_dataloader, desc="train", total=self.epoch_len)
        ):
            batch = self.process_batch(
                batch,
                self.train_metrics,
                next(self.reference_dataloader),
            )
            self.train_metrics.update("grad_norm", self._get_grad_norm(self.model))
            # log current results
            if self.ema_step and batch_idx % self.ema_step == 0:
                self.eema.update()
                self.fema.update()
                self.gema.update()
            if batch_idx % self.log_step == 0:
                self.writer.set_step((epoch - 1) * self.epoch_len + batch_idx)
                self.logger.debug(
                    "Train Epoch: {} {} GLoss: {:.6f} DLoss: {:.6f}".format(
                        epoch,
                        self._progress(batch_idx),
                        batch["generator_loss"],
                        batch["discriminator_loss"],
                    )
                )
                for met in self.metrics["train"]:
                    self.train_metrics.update(met.name, met.compute())
                self._log_scalars(self.train_metrics)
                self._log_batch(batch_idx, batch)
                # we don't want to reset train metrics at the start of every epoch
                # because we are interested in recent train metrics
                last_train_metrics = self.train_metrics.result()
                self.train_metrics.reset()
            if batch_idx + 1 >= self.epoch_len:
                break
        logs = last_train_metrics

        # Run val/test
        val_logs = self._evaluation_epoch(
            epoch, "evaluation", self.evaluation_dataloader
        )
        logs.update(**{f"evaluation_{name}": value for name, value in val_logs.items()})

        return logs

    def _evaluation_epoch(self, epoch, part, dataloader):
        """
        Evaluate model on the partition after training for an epoch.

        Args:
            epoch (int): current training epoch.
            part (str): partition to evaluate on
            dataloader (DataLoader): dataloader for the partition.
        Returns:
            logs (dict): logs that contain the information about evaluation.
        """
        if self.ema_step:
            self.eema.apply()
            self.fema.apply()
            self.gema.apply()
        self.is_train = False
        self.model.eval()
        self.evaluation_metrics.reset()
        with torch.no_grad():
            for batch_idx, batch in tqdm(
                enumerate(dataloader),
                desc=part,
                total=len(dataloader),
            ):
                batch = self.process_batch(
                    batch,
                    metrics=self.evaluation_metrics,
                )
            self.writer.set_step(epoch * self.epoch_len, part)
            for met in self.metrics["eval"]:
                self.evaluation_metrics.update(met.name, met.compute())
            self._log_scalars(self.evaluation_metrics)
            self._log_batch(
                batch_idx, batch, part
            )  # log only the last batch during inference
        if self.ema_step:
            self.gema.restore()
            self.fema.restore()
            self.eema.restore()
        return self.evaluation_metrics.result()

    @torch.no_grad()
    def _get_grad_norm(self, model, norm_type=2):
        """
        Calculates the gradient norm for logging.

        Args:
            norm_type (float | str | None): the order of the norm.
        Returns:
            total_norm (float): the calculated norm.
        """
        parameters = model.parameters()
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = [p for p in parameters if p.grad is not None]
        total_norm = torch.norm(
            torch.stack([torch.norm(p.grad.detach(), norm_type) for p in parameters]),
            norm_type,
        )
        return total_norm.item()

    def _progress(self, batch_idx):
        """
        Calculates the percentage of processed batch within the epoch.

        Args:
            batch_idx (int): the current batch index.
        Returns:
            progress (str): contains current step and percentage
                within the epoch.
        """
        base = "[{}/{} ({:.0f}%)]"
        if hasattr(self.train_dataloader, "n_samples"):
            current = batch_idx * self.train_dataloader.batch_size
            total = self.train_dataloader.n_samples
        else:
            current = batch_idx
            total = self.epoch_len
        return base.format(current, total, 100.0 * current / total)

    @abstractmethod
    def _log_batch(self, batch_idx, batch, mode="train"):
        """
        Abstract method. Should be defined in the nested Trainer Class.

        Log data from batch. Calls self.writer.add_* to log data
        to the experiment tracker.

        Args:
            batch_idx (int): index of the current batch.
            batch (dict): dict-based batch after going through
                the 'process_batch' function.
            mode (str): train or inference. Defines which logging
                rules to apply.
        """
        return NotImplementedError()

    def _log_scalars(self, metric_tracker: MetricTracker):
        """
        Wrapper around the writer 'add_scalar' to log all metrics.

        Args:
            metric_tracker (MetricTracker): calculated metrics.
        """
        if self.writer is None:
            return
        for metric_name in metric_tracker.keys():
            self.writer.add_scalar(f"{metric_name}", metric_tracker.avg(metric_name))

    def _save_checkpoint(self, epoch, save_best=False, only_best=False):
        """
        Save the checkpoints.

        Args:
            epoch (int): current epoch number.
            save_best (bool): if True, rename the saved checkpoint to 'model_best.pth'.
            only_best (bool): if True and the checkpoint is the best, save it only as
                'model_best.pth'(do not duplicate the checkpoint as
                checkpoint-epochEpochNumber.pth)
        """
        arch = type(self.model).__name__
        state = {
            "arch": arch,
            "epoch": epoch,
            "state_dict": self.model.state_dict(),
            "optimizer": {
                optimizer_name: optimizer.state_dict()
                for optimizer_name, optimizer in self.optimizer.items()
            },
            "config": self.config,
        }
        if self.ema_step is not None:
            state.update(
                {
                    "eema_weights": self.eema.ema_weights,
                    "gema_weights": self.gema.ema_weights,
                    "fema_weights": self.fema.ema_weights,
                }
            )
        filename = str(self.checkpoint_dir / f"checkpoint-epoch{epoch}.pth")
        if not (only_best and save_best):
            torch.save(state, filename)
            if self.config.writer.log_checkpoints:
                self.writer.add_checkpoint(filename, str(self.checkpoint_dir.parent))
            self.logger.info(f"Saving checkpoint: {filename} ...")
        if save_best:
            best_path = str(self.checkpoint_dir / "model_best.pth")
            torch.save(state, best_path)
            if self.config.writer.log_checkpoints:
                self.writer.add_checkpoint(best_path, str(self.checkpoint_dir.parent))
            self.logger.info("Saving current best: model_best.pth ...")

    def _resume_checkpoint(self, resume_path):
        """
        Resume from a saved checkpoint (in case of server crash, etc.).
        The function loads state dicts for everything, including model,
        optimizers, etc.

        Notice that the checkpoint should be located in the current experiment
        saved directory (where all checkpoints are saved in '_save_checkpoint').

        Args:
            resume_path (str): Path to the checkpoint to be resumed.
        """
        resume_path = str(resume_path)
        self.logger.info(f"Loading checkpoint: {resume_path} ...")
        checkpoint = torch.load(resume_path, self.device, weights_only=False)
        self.start_epoch = checkpoint["epoch"] + 1

        # load architecture params from checkpoint.
        if checkpoint["config"]["model"] != self.config["model"]:
            self.logger.warning(
                "Warning: Architecture configuration given in the config file is different from that "
                "of the checkpoint. This may yield an exception when state_dict is loaded."
            )
        self.model.load_state_dict(checkpoint["state_dict"])

        # load optimizer state from checkpoint only when optimizer type is not changed.
        if (
            checkpoint["config"]["optimizer"]
            != self.config["optimizer"]
            # or checkpoint["config"]["lr_scheduler"] != self.config["lr_scheduler"]
        ):
            self.logger.warning(
                "Warning: Optimizer or lr_scheduler given in the config file is different "
                "from that of the checkpoint. Optimizer and scheduler parameters "
                "are not resumed."
            )
        else:
            for optim_name in self.optimizer.keys():
                self.optimizer[optim_name].load_state_dict(
                    checkpoint["optimizer"][optim_name]
                )
        if self.ema_step is not None:
            if checkpoint.get("gema_weights", None) is None:
                self.gema.from_pretrained(checkpoint["epoch"])
                self.fema.from_pretrained(checkpoint["epoch"])
                self.eema.from_pretrained(checkpoint["epoch"])
            else:
                self.gema.from_pretrained(
                    checkpoint["epoch"], checkpoint["gema_weights"]
                )
                self.fema.from_pretrained(
                    checkpoint["epoch"], checkpoint["fema_weights"]
                )
                self.eema.from_pretrained(
                    checkpoint["epoch"], checkpoint["eema_weights"]
                )

        self.logger.info(
            f"Checkpoint loaded. Resume training from epoch {self.start_epoch}"
        )

    def _from_pretrained(self, pretrained_path):
        """
        Init model with weights from pretrained pth file.

        Notice that 'pretrained_path' can be any path on the disk. It is not
        necessary to locate it in the experiment saved dir. The function
        initializes only the model.

        Args:
            pretrained_path (str): path to the model state dict.
        """
        pretrained_path = str(pretrained_path)
        if hasattr(self, "logger"):  # to support both trainer and inferencer
            self.logger.info(f"Loading model weights from: {pretrained_path} ...")
        else:
            print(f"Loading model weights from: {pretrained_path} ...")
        checkpoint = torch.load(pretrained_path, self.device)

        if checkpoint.get("state_dict") is not None:
            self.model.load_state_dict(checkpoint["state_dict"])
        else:
            self.model.load_state_dict(checkpoint)

import numpy as np
import torch

from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer


class Trainer(BaseTrainer):
    """
    Trainer class. Defines the logic of batch logging and processing.
    """

    def clear_optimizer(self):
        for key in self.optimizer.keys():
            self.optimizer[key].zero_grad(set_to_none=True)

    def discriminator_step(
        self, real_images, real_labels, target_labels, latent_code=None, ref_image=None
    ):
        d_optimizer = self.optimizer["d_optimizer"]
        real_images = real_images.detach().requires_grad_(True)
        loss = []

        self.clear_optimizer()
        # sample from q dist
        with torch.no_grad():
            if ref_image is None:
                styles_embeddngs = self.model.mapping_network(
                    latent_code, target_labels
                )
            else:
                styles_embeddngs = self.model.style_encoder(ref_image, target_labels)
        fake_images = (
            self.model.generator(real_images, styles_embeddngs)
            .detach()
            .requires_grad_(True)
        )

        # compute loss
        real_preds = self.model.discriminator(real_images, real_labels)
        fake_preds = self.model.discriminator(fake_images, target_labels)

        d_loss = self.criterion["discriminator_loss"](
            fake_preds, real_preds, real_images, fake_images
        )

        d_loss.backward()
        d_optimizer.step()
        loss.append(d_loss.detach().item())
        return np.mean(loss)

    def generator_step(
        self, real_images, real_labels, target_labels, latent_code=None, ref_image=None
    ):
        g_optimizer = self.optimizer["g_optimizer"]
        e_optimizer = self.optimizer["e_optimizer"]
        f_optimizer = self.optimizer["f_optimizer"]
        self.clear_optimizer()
        real_images = real_images.detach()

        if ref_image is None:
            styles_embeddings1 = self.model.mapping_network(
                latent_code[0], target_labels
            )
            styles_embeddings2 = self.model.mapping_network(
                latent_code[0], target_labels
            )
        else:
            styles_embeddings1 = self.model.style_encoder(ref_image[0], target_labels)
            styles_embeddings2 = self.model.style_encoder(ref_image[1], target_labels)

        # adversarial
        fake_images = self.model.generator(real_images, styles_embeddings1)
        fake_preds = self.model.discriminator(fake_images, target_labels)
        real_preds = self.model.discriminator(real_images, real_labels)

        # style reconstruction (from generated image reconstruct styles_embeddngs)
        styles_reconstruction = self.model.style_encoder(fake_images, target_labels)

        # style diverse
        fake_images2 = self.model.generator(real_images, styles_embeddings2).detach()

        # cycle
        real_styles = self.model.style_encoder(real_images, real_labels)
        cycle_reconstruction = self.model.generator(fake_images, real_styles)

        g_loss = self.criterion["generator_loss"](
            fake_preds,
            styles_embeddings1,
            styles_reconstruction,
            fake_images,
            fake_images2,
            real_images,
            cycle_reconstruction,
            real_preds,
        )
        g_loss.backward()
        g_optimizer.step()
        if ref_image is None:
            f_optimizer.step()
            e_optimizer.step()
        return g_loss.detach().item(), fake_images.detach()

    def process_batch(
        self,
        batch,
        metrics: MetricTracker,
        reference_batch: dict[str : torch.Tensor] | None = None,
    ):
        """
        Run batch through the model, compute metrics, compute loss,
        and do training step (during training stage).

        The function expects that criterion aggregates all losses
        (if there are many) into a single one defined in the 'loss' key.

        Args:
            batch (dict): dict-based batch containing the data from
                the dataloader.
            metrics (MetricTracker): MetricTracker object that computes
                and aggregates the metrics. The metrics depend on the type of
                the partition (train or inference).
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform),
                model outputs, and losses.
        """

        metric_funcs = self.metrics["eval"]
        real_images = batch["data_object"].to(self.device)
        real_labels = batch["labels"].to(self.device)
        target_labels = torch.randint(0, self.config.num_domains, real_labels.size())
        latent_code1 = torch.randn(
            (real_images.size(0), self.config.latent_code_size),
            device=real_images.device,
        )
        latent_code2 = torch.randn(
            (real_images.size(0), self.config.latent_code_size),
            device=real_images.device,
        )

        if self.is_train:
            metric_funcs = self.metrics["train"]
            if reference_batch is not None:
                ref_images1 = reference_batch["data_object"].to(self.device)
                ref_images2 = reference_batch["reference_object"].to(self.device)
                target_labels = reference_batch["labels"].to(self.device)

            d_loss1 = self.discriminator_step(
                real_images, real_labels, target_labels, latent_code=latent_code1
            )
            self.train_metrics.update(
                "d_grad_norm", self._get_grad_norm(self.model.discriminator)
            )
            g_loss1, fake_images = self.generator_step(
                real_images,
                real_labels,
                target_labels,
                latent_code=[latent_code1, latent_code2],
            )
            verbose_losses = {
                "g_adversarial_loss": self.criterion["generator_loss"]
                .adversarial.loss_value.detach()
                .item(),
                "g_style_reconstruction_loss": self.criterion["generator_loss"]
                .style_reconstruction.loss_value.detach()
                .item(),
                "g_style_diversityn_loss": self.criterion["generator_loss"]
                .style_diversity.loss_value.detach()
                .item(),
                "g_cycle_loss": self.criterion["generator_loss"]
                .cycle.loss_value.detach()
                .item(),
                "d_adversarial_loss": self.criterion["discriminator_loss"]
                .adversarial.loss_value.detach()
                .item(),
            }
            self.train_metrics.update(
                "g_grad_norm", self._get_grad_norm(self.model.generator)
            )
            self.train_metrics.update(
                "f_grad_norm", self._get_grad_norm(self.model.mapping_network)
            )
            self.train_metrics.update(
                "e_grad_norm", self._get_grad_norm(self.model.style_encoder)
            )
            if reference_batch is not None:
                d_loss2 = self.discriminator_step(
                    real_images, real_labels, target_labels, ref_image=ref_images1
                )
                self.train_metrics.update(
                    "d_grad_norm", self._get_grad_norm(self.model.discriminator)
                )
                g_loss2, _ = self.generator_step(
                    real_images,
                    real_labels,
                    target_labels,
                    ref_image=[ref_images1, ref_images2],
                )
                self.train_metrics.update(
                    "g_grad_norm", self._get_grad_norm(self.model.generator)
                )

            all_losses = {"generator_loss": g_loss1, "discriminator_loss": d_loss1}
            if reference_batch is not None:
                all_losses["generator_loss"] = np.mean(
                    (all_losses["generator_loss"], g_loss2)
                )
                all_losses["discriminator_loss"] = np.mean(
                    (all_losses["discriminator_loss"], d_loss2)
                )
                all_losses["generator_loss"] = np.mean(
                    (all_losses["generator_loss"], g_loss2)
                )
                all_losses["discriminator_loss"] = np.mean(
                    (all_losses["discriminator_loss"], d_loss2)
                )
                verbose_losses2 = {
                    "g_adversarial_loss": self.criterion["generator_loss"]
                    .adversarial.loss_value.detach()
                    .item(),
                    "g_style_reconstruction_loss": self.criterion["generator_loss"]
                    .style_reconstruction.loss_value.detach()
                    .item(),
                    "g_style_diversityn_loss": self.criterion["generator_loss"]
                    .style_diversity.loss_value.detach()
                    .item(),
                    "g_cycle_loss": self.criterion["generator_loss"]
                    .cycle.loss_value.detach()
                    .item(),
                    "d_adversarial_loss": self.criterion["discriminator_loss"]
                    .adversarial.loss_value.detach()
                    .item(),
                }
                for key in verbose_losses.keys():
                    verbose_losses[key] = np.mean(
                        (verbose_losses[key], verbose_losses2[key])
                    )

            all_losses.update(verbose_losses)
            batch.update(all_losses)

            # update metrics for each loss (in case of multiple losses)
            for loss_name in self.config.writer.loss_names:
                metrics.update(loss_name, batch[loss_name])
        else:
            style_embedding = self.model.mapping_network(latent_code1, target_labels)
            fake_images = self.model.generator(real_images, style_embedding)

        for met in metric_funcs:
            met(fake_images, real_images)
        fake_images = self.denormalize(fake_images)
        real_images = self.denormalize(real_images)

        batch.update(
            {
                "generated_images": fake_images,
                "target_labels": target_labels,
                "real_images": real_images.cpu(),
                "real_labels": real_labels.cpu(),
            }
        )
        return batch

    def denormalize(self, imgs):
        imgs = (imgs + 1) / 2
        imgs = (imgs * 255).clamp(0, 255).to(torch.uint8)
        return imgs

    def _log_batch(self, batch_idx, batch, mode="train"):
        """
        Log data from batch. Calls self.writer.add_* to log data
        to the experiment tracker.

        Args:
            batch_idx (int): index of the current batch.
            batch (dict): dict-based batch after going through
                the 'process_batch' function.
            mode (str): train or inference. Defines which logging
                rules to apply.
        """
        # method to log data from you batch
        # such as audio, text or images, for example

        # logging scheme might be different for different partitions
        if mode == "train":  # the method is called only every self.log_step steps
            self.writer.add_image(batch)
        else:
            self.writer.add_image(batch)

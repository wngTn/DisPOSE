from lightning.pytorch.callbacks import RichProgressBar


class SmartRichProgressBar(RichProgressBar):
    """RichProgressBar with step-based progress and smart metric formatting."""

    def _get_train_description(self, current_epoch: int) -> str:
        if self.trainer.max_steps and self.trainer.max_steps > 0:
            return "Training"
        return super()._get_train_description(current_epoch)

    def on_train_epoch_start(self, trainer, pl_module):
        # For step-based training, show a single bar from 0 to max_steps
        # instead of resetting per epoch.
        if not (trainer.max_steps and trainer.max_steps > 0):
            return super().on_train_epoch_start(trainer, pl_module)

        if self.is_disabled:
            return

        total = trainer.max_steps
        description = self._get_train_description(trainer.current_epoch)

        if self.train_progress_bar_id is not None and self._leave:
            self._stop_progress()
            self._init_progress(trainer)

        if self.progress is not None:
            if self.train_progress_bar_id is None:
                self.train_progress_bar_id = self._add_task(total, description)
                self.progress.update(self.train_progress_bar_id, completed=trainer.global_step)
            else:
                self.progress.reset(
                    self.train_progress_bar_id,
                    total=total,
                    completed=trainer.global_step,
                    description=(
                        f"[{self.theme.description}]{description}"
                        if self.theme.description
                        else description
                    ),
                    visible=True,
                )
        self.refresh()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not (trainer.max_steps and trainer.max_steps > 0):
            return super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)

        if not self.is_disabled and self.train_progress_bar_id is None:
            self._initialize_train_progress_bar_id()

        self._update(self.train_progress_bar_id, trainer.global_step + 1)
        self._update_metrics(trainer, pl_module)
        self.refresh()

    def get_metrics(self, trainer, pl_module):
        metrics = super().get_metrics(trainer, pl_module)
        # Remove v_num — it's just the W&B run ID and not informative
        metrics.pop("v_num", None)
        # Auto-format: scientific notation for small values, 4 decimals otherwise
        formatted = {}
        for k, v in metrics.items():
            if isinstance(v, float):
                formatted[k] = f"{v:.3e}" if (0 < abs(v) < 0.01) else f"{v:.4f}"
            else:
                formatted[k] = v
        return formatted

from abc import abstractmethod
import os
import time
import json

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
import numpy as np

from torchvision import utils
from torch.utils.tensorboard import SummaryWriter

from .utils import *
from ..utils.general_utils import *
from ..utils.data_utils import recursive_to_device, cycle, ResumableSampler


class Trainer:
    """
    Base class for training.
    """
    def __init__(self,
        models,
        dataset,
        *,
        output_dir,
        load_dir,
        step,
        max_steps,
        batch_size=None,
        batch_size_per_gpu=None,
        batch_split=None,
        optimizer={},
        lr_scheduler=None,
        elastic=None,
        grad_clip=None,
        ema_rate=0.9999,
        fp16_mode='inflat_all',
        fp16_scale_growth=1e-3,
        finetune_ckpt=None,
        log_param_stats=False,
        prefetch_data=True,
        i_print=1000,
        i_log=500,
        i_sample=10000,
        i_save=10000,
        i_ddpcheck=10000,
        reset_optimizer=False,
        **kwargs
    ):
        assert batch_size is not None or batch_size_per_gpu is not None, 'Either batch_size or batch_size_per_gpu must be specified.'

        self.models = models
        self.dataset = dataset
        self.batch_split = batch_split if batch_split is not None else 1
        self.max_steps = max_steps
        self.optimizer_config = optimizer
        self.lr_scheduler_config = lr_scheduler
        self.elastic_controller_config = elastic
        self.grad_clip = grad_clip
        self.ema_rate = [ema_rate] if isinstance(ema_rate, float) else ema_rate
        self.fp16_mode = fp16_mode
        self.fp16_scale_growth = fp16_scale_growth
        self.log_param_stats = log_param_stats
        self.prefetch_data = prefetch_data
        if self.prefetch_data:
            self._data_prefetched = None

        self.output_dir = output_dir
        self.i_print = i_print
        self.i_log = i_log
        self.i_sample = i_sample
        self.i_save = i_save
        self.i_ddpcheck = i_ddpcheck        

        if dist.is_initialized():
            # Multi-GPU params
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
            self.local_rank = dist.get_rank() % torch.cuda.device_count()
            self.is_master = self.rank == 0
        else:
            # Single-GPU params
            self.world_size = 1
            self.rank = 0
            self.local_rank = 0
            self.is_master = True

        self.batch_size = batch_size if batch_size_per_gpu is None else batch_size_per_gpu * self.world_size
        self.batch_size_per_gpu = batch_size_per_gpu if batch_size_per_gpu is not None else batch_size // self.world_size
        assert self.batch_size % self.world_size == 0, 'Batch size must be divisible by the number of GPUs.'
        assert self.batch_size_per_gpu % self.batch_split == 0, 'Batch size per GPU must be divisible by batch split.'

        self.init_models_and_more(**kwargs)
        self.prepare_dataloader(**kwargs)
        
        # Load checkpoint
        self.step = 0
        self.reset_optimizer = reset_optimizer
        if load_dir is not None and step is not None:
            self.load(load_dir, step, reset_optimizer=reset_optimizer)
        elif finetune_ckpt is not None:
            self.finetune_from(finetune_ckpt)

        if self.is_master:
            os.makedirs(os.path.join(self.output_dir, 'ckpts'), exist_ok=True)
            os.makedirs(os.path.join(self.output_dir, 'samples'), exist_ok=True)
            self.writer = SummaryWriter(os.path.join(self.output_dir, 'tb_logs'))
            self.use_wandb = False

            try:
                import wandb
                _has_wandb = True
            except ImportError:
                _has_wandb = False
            if _has_wandb and os.environ.get('WANDB_API_KEY'):
                try:
                    wandb.init(
                        project=os.environ.get('WANDB_PROJECT', 'trellis'),
                        name=os.environ.get('WANDB_RUN_NAME', os.path.basename(self.output_dir)),
                        dir=self.output_dir,
                        config={
                            'batch_size': self.batch_size,
                            'batch_size_per_gpu': self.batch_size_per_gpu,
                            'batch_split': self.batch_split,
                            'max_steps': self.max_steps,
                            'fp16_mode': self.fp16_mode,
                            'ema_rate': self.ema_rate,
                        },
                        resume='allow',
                    )
                    self.use_wandb = True
                    print(f'[wandb] Initialized: {wandb.run.url}')
                except Exception as e:
                    print(f'[wandb] Failed to initialize: {e}')
            else:
                print(f'[wandb] Skipped (has_wandb={_has_wandb}, api_key_set={bool(os.environ.get("WANDB_API_KEY"))})')

        if self.world_size > 1:
            self.check_ddp()
            
        if self.is_master:
            print('\n\nTrainer initialized.')
            print(self)
            
    @property
    def device(self):
        for _, model in self.models.items():
            if hasattr(model, 'device'):
                return model.device
        return next(list(self.models.values())[0].parameters()).device
            
    @abstractmethod
    def init_models_and_more(self, **kwargs):
        """
        Initialize models and more.
        """
        pass
    
    def prepare_dataloader(self, **kwargs):
        """
        Prepare dataloader.
        """
        self.data_sampler = ResumableSampler(
            self.dataset,
            shuffle=True,
        )
        pin_memory = getattr(self.dataset, 'PIN_MEMORY', True)
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size_per_gpu,
            num_workers=int(np.ceil(os.cpu_count() / torch.cuda.device_count())),
            pin_memory=pin_memory,
            drop_last=True,
            persistent_workers=True,
            collate_fn=self.dataset.collate_fn if hasattr(self.dataset, 'collate_fn') else None,
            sampler=self.data_sampler,
        )
        self.data_iterator = cycle(self.dataloader)

    @abstractmethod
    def load(self, load_dir, step=0):
        """
        Load a checkpoint.
        Should be called by all processes.
        """
        pass

    @abstractmethod
    def save(self):
        """
        Save a checkpoint.
        Should be called only by the rank 0 process.
        """
        pass
    
    @abstractmethod
    def finetune_from(self, finetune_ckpt):
        """
        Finetune from a checkpoint.
        Should be called by all processes.
        """
        pass
    
    @abstractmethod
    def run_snapshot(self, num_samples, batch_size=4, verbose=False, **kwargs):
        """
        Run a snapshot of the model.
        """
        pass

    @torch.no_grad()
    def visualize_sample(self, sample):
        """
        Convert a sample to an image.
        """
        if hasattr(self.dataset, 'visualize_sample'):
            return self.dataset.visualize_sample(sample)
        else:
            return sample

    @torch.no_grad()
    def snapshot_dataset(self, num_samples=100):
        """
        Sample images from the dataset.
        """
        dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=num_samples,
            num_workers=0,
            shuffle=True,
            collate_fn=self.dataset.collate_fn if hasattr(self.dataset, 'collate_fn') else None,
        )
        data = next(iter(dataloader))
        data = recursive_to_device(data, self.device)
        vis = self.visualize_sample(data)
        if isinstance(vis, dict):
            save_cfg = [(f'dataset_{k}', v) for k, v in vis.items()]
        else:
            save_cfg = [('dataset', vis)]
        for name, image in save_cfg:
            utils.save_image(
                image,
                os.path.join(self.output_dir, 'samples', f'{name}.jpg'),
                nrow=int(np.sqrt(num_samples)),
                normalize=True,
                value_range=self.dataset.value_range,
            )

    @torch.no_grad()
    def snapshot(self, suffix=None, num_samples=4, batch_size=4, verbose=False):
        """
        Sample images from the model.
        NOTE: This function should be called by all processes.
        """
        if self.is_master:
            print(f'\nSampling {num_samples} images...', end='')

        if suffix is None:
            suffix = f'step{self.step:07d}'

        # Assign tasks
        num_samples_per_process = int(np.ceil(num_samples / self.world_size))
        samples = self.run_snapshot(num_samples_per_process, batch_size=batch_size, verbose=verbose)

        # Preprocess images
        for key in list(samples.keys()):
            if samples[key]['type'] == 'sample':
                vis = self.visualize_sample(samples[key]['value'])
                if isinstance(vis, dict):
                    for k, v in vis.items():
                        samples[f'{key}_{k}'] = {'value': v, 'type': 'image'}
                    del samples[key]
                else:
                    samples[key] = {'value': vis, 'type': 'image'}

        # Gather results
        if self.world_size > 1:
            for key in samples.keys():
                v = samples[key]['value']
                # 'scalar' / 'scalar_list' values are Python numbers (computed
                # identically on every rank because run_snapshot iterates the
                # full val set on each rank). Skip the tensor gather for them
                # and just keep rank 0's value for wandb logging downstream.
                if not isinstance(v, torch.Tensor):
                    continue
                samples[key]['value'] = v.contiguous()
                if self.is_master:
                    all_images = [torch.empty_like(samples[key]['value']) for _ in range(self.world_size)]
                else:
                    all_images = []
                dist.gather(samples[key]['value'], all_images, dst=0)
                if self.is_master:
                    samples[key]['value'] = torch.cat(all_images, dim=0)[:num_samples]

        # Save images
        if self.is_master:
            os.makedirs(os.path.join(self.output_dir, 'samples', suffix), exist_ok=True)
            for key in samples.keys():
                if samples[key]['type'] == 'image':
                    utils.save_image(
                        samples[key]['value'],
                        os.path.join(self.output_dir, 'samples', suffix, f'{key}_{suffix}.jpg'),
                        nrow=int(np.sqrt(num_samples)),
                        normalize=True,
                        value_range=self.dataset.value_range,
                    )
                elif samples[key]['type'] == 'number':
                    min = samples[key]['value'].min()
                    max = samples[key]['value'].max()
                    images = (samples[key]['value'] - min) / (max - min)
                    images = utils.make_grid(
                        images,
                        nrow=int(np.sqrt(num_samples)),
                        normalize=False,
                    )
                    save_image_with_notes(
                        images,
                        os.path.join(self.output_dir, 'samples', suffix, f'{key}_{suffix}.jpg'),
                        notes=f'{key} min: {min}, max: {max}',
                    )

        # Log snapshot to wandb
        if self.is_master and self.use_wandb:
            import wandb
            wandb_log = {}
            for key in samples.keys():
                if samples[key]['type'] == 'image':
                    img_path = os.path.join(self.output_dir, 'samples', suffix, f'{key}_{suffix}.jpg')
                    if os.path.exists(img_path):
                        wandb_log[f'snapshot/{key}'] = wandb.Image(img_path, caption=f'{key} @ step {self.step}')
                elif samples[key]['type'] == 'scalar':
                    wandb_log[f'val/{key}'] = samples[key]['value']
                elif samples[key]['type'] == 'scalar_list':
                    for i, v in enumerate(samples[key]['value']):
                        wandb_log[f'val/{key}/frame_{i}'] = v
            if wandb_log:
                wandb.log(wandb_log, step=self.step)

        if self.is_master:
            print(' Done.')

    @abstractmethod
    def update_ema(self):
        """
        Update exponential moving average.
        Should only be called by the rank 0 process.
        """
        pass

    @abstractmethod
    def check_ddp(self):
        """
        Check if DDP is working properly.
        Should be called by all process.
        """
        pass

    @abstractmethod
    def training_losses(**mb_data):
        """
        Compute training losses.
        """
        pass
    
    def load_data(self):
        """
        Load data.
        """
        if self.prefetch_data:
            if self._data_prefetched is None:
                self._data_prefetched = recursive_to_device(next(self.data_iterator), self.device, non_blocking=True)
            data = self._data_prefetched
            self._data_prefetched = recursive_to_device(next(self.data_iterator), self.device, non_blocking=True)
        else:
            data = recursive_to_device(next(self.data_iterator), self.device, non_blocking=True)
        
        # if the data is a dict, we need to split it into multiple dicts with batch_size_per_gpu
        if isinstance(data, dict):
            if self.batch_split == 1:
                data_list = [data]
            else:
                batch_size = list(data.values())[0].shape[0]
                data_list = [
                    {k: v[i * batch_size // self.batch_split:(i + 1) * batch_size // self.batch_split] for k, v in data.items()}
                    for i in range(self.batch_split)
                ]
        elif isinstance(data, list):
            data_list = data
        else:
            raise ValueError('Data must be a dict or a list of dicts.')
        
        return data_list

    @abstractmethod
    def run_step(self, data_list):
        """
        Run a training step.
        """
        pass

    def run(self):
        """
        Run training.
        """
        if self.is_master:
            print('\nStarting training...')
            print('[skip snapshot_dataset for sequential data]')
        if self.is_master:
            print('Skipping initial/resume snapshot, starting training directly.')

        log = []
        time_last_print = 0.0
        time_elapsed = 0.0
        while self.step < self.max_steps:
            time_start = time.time()

            if self.step % 100 == 0 and self.is_master:
                print(f'[step {self.step}] loading data...', flush=True)
            data_list = self.load_data()
            if self.step % 100 == 0 and self.is_master:
                print(f'[step {self.step}] running step...', flush=True)
            step_log = self.run_step(data_list)

            time_end = time.time()
            time_elapsed += time_end - time_start

            self.step += 1

            # Print progress
            if self.is_master and self.step % self.i_print == 0:
                speed = self.i_print / (time_elapsed - time_last_print) * 3600
                columns = [
                    f'Step: {self.step}/{self.max_steps} ({self.step / self.max_steps * 100:.2f}%)',
                    f'Elapsed: {time_elapsed / 3600:.2f} h',
                    f'Speed: {speed:.2f} steps/h',
                    f'ETA: {(self.max_steps - self.step) / speed:.2f} h',
                ]
                print(' | '.join([c.ljust(25) for c in columns]), flush=True)
                time_last_print = time_elapsed

            # Check ddp
            if self.world_size > 1 and self.i_ddpcheck is not None and self.step % self.i_ddpcheck == 0:
                self.check_ddp()

            if self.is_master and self.step % self.i_save == 0:                                                
                self.save()
    
            # Sample images
            if self.step % self.i_sample == 0:
                self.snapshot()

            if self.is_master:
                log.append((self.step, {}))

                # Log time
                log[-1][1]['time'] = {
                    'step': time_end - time_start,
                    'elapsed': time_elapsed,
                }

                # Log losses
                if step_log is not None:
                    log[-1][1].update(step_log)

                # Log scale
                if self.fp16_mode == 'amp':
                    log[-1][1]['scale'] = self.scaler.get_scale()
                elif self.fp16_mode == 'inflat_all':
                    log[-1][1]['log_scale'] = self.log_scale

                # Save log
                if self.step % self.i_log == 0:
                    ## save to log file
                    def _json_default(obj):
                        if isinstance(obj, (np.floating, np.integer)):
                            return float(obj)
                        raise TypeError(f'Object of type {type(obj).__name__} is not JSON serializable')
                    log_str = '\n'.join([
                        f'{step}: {json.dumps(log, default=_json_default)}' for step, log in log
                    ])
                    with open(os.path.join(self.output_dir, 'log.txt'), 'a') as log_file:
                        log_file.write(log_str + '\n')

                    # show with mlflow
                    log_show = [l for _, l in log if not dict_any(l, lambda x: np.isnan(x))]
                    log_show = dict_reduce(log_show, lambda x: np.mean(x))
                    log_show = dict_flatten(log_show, sep='/')
                    for key, value in log_show.items():
                        self.writer.add_scalar(key, value, self.step)
                    if self.use_wandb:
                        import wandb
                        wandb.log(log_show, step=self.step)
                    log = []

        # Final snapshot must run on ALL ranks (like the periodic one at
        # i_sample) — run_snapshot may issue collectives (e.g. all_gather_object
        # in the sharded SS validation), which would hang if master-only.
        self.snapshot(suffix='final')
        if self.is_master:
            self.writer.close()
            if self.use_wandb:
                import wandb
                wandb.finish()
            print('Training finished.')
            
    def profile(self, wait=2, warmup=3, active=5):
        """
        Profile the training loop.
        """
        with torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(os.path.join(self.output_dir, 'profile')),
            profile_memory=True,
            with_stack=True,
        ) as prof:
            for _ in range(wait + warmup + active):
                self.run_step()
                prof.step()
            
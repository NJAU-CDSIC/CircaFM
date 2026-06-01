import argparse
from argparse import Namespace
import random
import numpy as np
import os 
import torch
from data_provider.classfication_datasets import MulGroup_MultipleDataset
from torch.utils.data import DataLoader
from models.circaLLM import CIRCALLM
from peft import LoraConfig, get_peft_model, PeftModel
from tqdm import tqdm 
from utils.metrics import Metric
from tabulate import tabulate
from datetime import datetime



class Circadian_Trainer:
    def __init__(self, args: Namespace):
        self.args = args
        self.epoch = 0
        self.trainAddr = os.path.join(self.args.assets_path, "train_results/")
        self.valAddr = os.path.join(self.args.assets_path, "val_results/")
        self.testAddr = os.path.join(self.args.assets_path, "test_results/")
        self.patience = getattr(args, 'early_stop_patience', 3)
        self.min_delta = getattr(args, 'min_delta', 1e-4)
        current_time = datetime.now()
        self.c_time = current_time.strftime("%y_%m_%d_%H_%M")

        # 1. Model configuration
        self.config = self._init_model_config()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config.device = self.device.type
        print(f"Using device: {self.device.type}")

        # 2. Load pretrained model
        self.model = self._init_and_load_pretrained_model()

        # 3. Freeze/unfreeze strategy
        self._freeze_base_model_unfreeze_head()

        # 4. Initialize dataset, loss, optimizer
        self._init_dataset()
        self._init_loss_optimizer()

    def _init_model_config(self):
        config_dict = {
            "task_name": "diffrhythm", 
            "model_name": "CIRCALLM", 
            "transformer_type": "encoder_only", 
            "freeze_embedder": False,
            "freeze_encoder": False,
            "freeze_head": False,
            "learning_rate": 1e-6,
            "num_epochs": 30,
            "n_channels": 2,
            "n_classes_head1": 2,
            "n_classes_head2": 2,
            "reduction": self.args.reduction,
            "d_model": None, 
            "seq_len": self.args.seq_len,
            "enable_gradient_checkpointing": False,
            "enable_FAN": True,
            "enable_FAN_gate": True,
            "patch_len": 6, 
            "patch_stride_len": 6, 
            "device": "cpu",
            "transformer_backbone": "google/flan-t5-small", 
            "model_kwargs": {},
            "t5_config": {
                "architectures": ["T5ForConditionalGeneration"],
                "d_ff": 1024,
                "d_kv": 64,
                "d_model": 512,
                "decoder_start_token_id": 0,
                "dropout_rate": 0.1,
                "eos_token_id": 1,
                "feed_forward_proj": "gelu",
                "initializer_factor": 1.0,
                "is_encoder_decoder": True,
                "layer_norm_epsilon": 1e-06,
                "model_type": "t5",
                "n_positions": self.args.seq_len,
                "num_decoder_layers": 6,
                "num_heads": 8,
                "num_layers": 6,
                "output_past": True,
                "pad_token_id": 0,
                "relative_attention_max_distance": 128,
                "relative_attention_num_buckets": 32,
                "tie_word_embeddings": False,
                "use_cache": True,
                "vocab_size": 32128,
            }
        }
        return Namespace(**config_dict)

    def _verify_loaded_weights(self):
        param_values = []
        for param in self.model.parameters():
            if param.requires_grad:
                param_values.append(param.detach().cpu().numpy().flatten())
        if not param_values:
            print("⚠️  No trainable parameters found in the model!")
            return
        
        all_params = np.concatenate(param_values)
        mean_val = np.mean(all_params)
        std_val = np.std(all_params)
        print(f"Trainable parameter statistics: mean={mean_val:.6f}, std={std_val:.6f}")

        if abs(mean_val) < 1e-4:
            print("❌ Weight loading may have failed! Trainable parameters are still near initialization.")
            for i, (name, param) in enumerate(self.model.named_parameters()):
                if param.requires_grad:
                    print(f"Parameter {name} first 5 values: {param.detach().cpu().numpy().flatten()[:5]}")
                    if i >= 2:
                        break
        else:
            print("✅ Weight loading successful! Trainable parameters are far from initialization.")

    def _init_and_load_pretrained_model(self):
        base_model = CIRCALLM(self.config)
        base_model.to(self.device)
        print("Base CIRCALLM model initialized")

        if not os.path.exists(self.args.pretrained_model_path):
            raise FileNotFoundError(f"Pretrained model path does not exist: {self.args.pretrained_model_path}")
        
        is_file = os.path.isfile(self.args.pretrained_model_path)
        is_dir = os.path.isdir(self.args.pretrained_model_path)
        print(f"Pretrained model path type: {'file' if is_file else 'folder'}")

        if is_file:
            checkpoint = torch.load(
                self.args.pretrained_model_path, 
                map_location=self.device, 
                weights_only=True
            )
            pretrained_state = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
            print(f"Loaded checkpoint with {len(pretrained_state)} parameter keys")

            is_lora_pretrained = any('lora_' in key for key in pretrained_state.keys())
            print(f"Pretrained model type: {'LoRA model' if is_lora_pretrained else 'base model'}")

            if is_lora_pretrained:
                lora_config = LoraConfig(
                    r=16,
                    lora_alpha=32,
                    target_modules=["q", "v"],
                    lora_dropout=0.1,
                    bias="none",
                    task_type="SEQ_CLS"
                )
                lora_model = get_peft_model(base_model, lora_config)
                lora_model.to(self.device)
                self.model = lora_model
                print("Added LoRA structure to base model")

                try:
                    load_result = lora_model.load_state_dict(pretrained_state, strict=True)
                    print("Strictly loaded LoRA model weights successfully")
                    if load_result.missing_keys:
                        print(f"  - Missing keys: {load_result.missing_keys[:3]}...")
                    else:
                        print("  - No missing keys")
                    if load_result.unexpected_keys:
                        print(f"  - Unexpected keys: {load_result.unexpected_keys[:3]}...")
                    else:
                        print("  - No unexpected keys")
                except RuntimeError as e:
                    print(f"Strict loading failed, falling back to non-strict: {str(e)[:300]}")
                    load_result = lora_model.load_state_dict(pretrained_state, strict=False)
                    self._log_param_mismatch(pretrained_state)
                
                model = lora_model

            else:
                corrected_state = {}
                self.model = base_model
                for key, value in pretrained_state.items():
                    if key.startswith('base_model.model.'):
                        corrected_key = key.replace('base_model.model.', '')
                    else:
                        corrected_key = key
                    corrected_state[corrected_key] = value

                try:
                    load_result = base_model.load_state_dict(corrected_state, strict=True)
                    print("Strictly loaded base model weights successfully")
                except RuntimeError as e:
                    print(f"Strict loading failed, falling back to non-strict: {str(e)[:300]}")
                    load_result = base_model.load_state_dict(corrected_state, strict=False)
                    self._log_param_mismatch(corrected_state)
                
                model = base_model

        elif is_dir:
            print(f"Loading LoRA adapter from folder: {self.args.pretrained_model_path}")
            lora_model = PeftModel.from_pretrained(
                model=base_model,
                model_id=self.args.pretrained_model_path,
                device_map=self.device,
                is_trainable=True
            )
            lora_model.to(self.device)
            self.model = lora_model
            print("✅ LoRA adapter folder loaded successfully")

        else:
            raise ValueError(f"Invalid pretrained model path: {self.args.pretrained_model_path}")

        self.model = model
        self._verify_loaded_weights()

        if self.args.lora and not isinstance(self.model, PeftModel):
            print("LoRA enabled for training, adding LoRA structure to base model")
            lora_config = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=["q", "v"],
                lora_dropout=0.1,
                bias="none"
            )
            self.model = get_peft_model(self.model, lora_config)
            self.model.to(self.device)
            self._verify_loaded_weights()

        return self.model

    def _log_param_mismatch(self, loaded_state):
        model_keys = set(self.model.state_dict().keys())
        loaded_keys = set(loaded_state.keys())
        missing_keys = model_keys - loaded_keys
        unexpected_keys = loaded_keys - model_keys
        if missing_keys:
            print(f"Model missing keys (first 5): {list(missing_keys)[:5]}...")
        if unexpected_keys:
            print(f"Weights with unexpected keys (first 5): {list(unexpected_keys)[:5]}...")

    def _freeze_base_model_unfreeze_head(self):
        if isinstance(self.model, PeftModel):
            for param in self.model.base_model.parameters():
                param.requires_grad = False
            for name, param in self.model.named_parameters():
                if "lora_" in name:
                    param.requires_grad = True
            head_params = self.model.base_model.model.head.parameters()
        else:
            for name, param in self.model.named_parameters():
                if "head" not in name:
                    param.requires_grad = False
            head_params = self.model.head.parameters()

        for param in head_params:
            param.requires_grad = True

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"Trainable parameters: {trainable:,} / Total: {total:,} ({100*trainable/total:.2f}%)")

    def _init_dataset(self):
        file_paths = self.get_filePaths(self.args.dataset_path)
        self.train_dataset = MulGroup_MultipleDataset(
            data_split="train", 
            file_paths=file_paths, 
            seq_len=self.args.seq_len, 
            normalize_method="separate"
        )
        self.val_dataset = MulGroup_MultipleDataset(
            data_split="val", 
            file_paths=file_paths, 
            seq_len=self.args.seq_len, 
            normalize_method="separate"
        )
        self.train_dataloader = DataLoader(
            self.train_dataset, 
            batch_size=self.args.batch_size, 
            shuffle=True
        )
        self.val_dataloader = DataLoader(
            self.val_dataset, 
            batch_size=self.args.batch_size, 
            shuffle=False
        )
        print(f"Dataset loaded: training {len(self.train_dataset)} samples, validation {len(self.val_dataset)} samples")

    def _init_loss_optimizer(self):
        label = self.train_dataset.labels[:, :]
        def safe_weight(counts):
            return counts[0] / (counts[1] + 1e-8) if len(counts) >= 2 else 1.0
        
        nn_period = safe_weight(np.unique(label[:, 0], return_counts=True)[1])
        nn_phase = safe_weight(np.unique(label[:, 1], return_counts=True)[1])
        nn_amp = safe_weight(np.unique(label[:, 2], return_counts=True)[1])
        nn_mesor = safe_weight(np.unique(label[:, 3], return_counts=True)[1])

        head1_pos_weights = torch.tensor([nn_period, nn_phase], dtype=torch.float32).to(self.device)
        head2_pos_weights = torch.tensor([nn_amp, nn_mesor], dtype=torch.float32).to(self.device)
        self.criterion_head1 = torch.nn.BCEWithLogitsLoss(pos_weight=head1_pos_weights)
        self.criterion_head2 = torch.nn.BCEWithLogitsLoss(pos_weight=head2_pos_weights)
        self.criterion_head1_2 = torch.nn.BCEWithLogitsLoss(pos_weight=head1_pos_weights, reduction='none')
        self.criterion_head2_2 = torch.nn.BCEWithLogitsLoss(pos_weight=head2_pos_weights, reduction='none')

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.init_lr)
        total_steps = self.args.epochs * len(self.train_dataloader)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer, 
            max_lr=self.args.max_lr, 
            total_steps=total_steps
        )
        print("Loss functions, optimizer, and scheduler initialized")

    def get_filePaths(self, folder_path):
        print(f"Current working directory: {os.getcwd()}")
        originFiles = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
        files = sorted(originFiles)
        file_paths = [os.path.join(folder_path, item, "") for item in files] 
        print(file_paths)
        return file_paths[:]

    def train_epoch(self):
        self.model.to(self.device)
        self.model.train()
        all_targets, all_preds, all_scores = [], [], []
        sample_metadata = []
        running_loss, t_running_loss, amp_running_loss, phase_running_loss, mesor_running_loss = 0.0, 0.0, 0.0, 0.0, 0.0
        correct, correct_T, correct_Amp, correct_Phase, correct_Mesor = 0, 0, 0, 0, 0
        total = 0
        
        for batch_data_1, input_mask_1, x_marks_1, batch_data_2, input_mask_2, x_marks_2, targets, batch_sample_ids, batch_sample_folders in tqdm(
            self.train_dataloader, total=len(self.train_dataloader), desc=f"Train Epoch {self.epoch}"
        ):
            self.optimizer.zero_grad()
            
            batch_data_1 = batch_data_1.to(self.device).float()
            batch_data_2 = batch_data_2.to(self.device).float()
            input_mask_1 = input_mask_1.long().to(self.device)
            input_mask_2 = input_mask_2.long().to(self.device)
            x_marks_1 = x_marks_1.to(self.device)
            x_marks_2 = x_marks_2.to(self.device)
            targets = targets.float().to(self.device)

            all_targets.extend(targets.detach().cpu().numpy())
            total += targets.size(0)

            dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8) else torch.float32
            with torch.autocast(device_type=self.device.type, dtype=dtype, enabled=True):
                output = self.model(
                    x_enc=batch_data_1,
                    input_mask=input_mask_1,
                    x_mark=x_marks_1,
                    x_enc2=batch_data_2,
                    input_mask2=input_mask_2,
                    x_mark2=x_marks_2,
                    reduction=self.args.reduction,
                    return_dict=True
                )
                
                combined_logits = output.logits
                metadata = output.metadata
                head1_classes = metadata['head1_classes']
                
                head1_logits = combined_logits[:, :head1_classes]
                head2_logits = combined_logits[:, head1_classes:]
                head1_targets = targets[:, :head1_classes]
                head2_targets = targets[:, head1_classes:]
                
                loss_head1 = self.criterion_head1(head1_logits, head1_targets)
                loss_head2 = self.criterion_head2(head2_logits, head2_targets)
                loss_head1_2 = self.criterion_head1_2(head1_logits, head1_targets)
                loss_head2_2 = self.criterion_head2_2(head2_logits, head2_targets)
                total_loss = loss_head1 + loss_head2

                period_loss = loss_head1_2[:, 0].mean()
                phase_loss = loss_head1_2[:, 1].mean()
                amp_loss = loss_head2_2[:, 0].mean()
                mesor_loss = loss_head2_2[:, 1].mean()

            total_loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            running_loss += total_loss.item() / 2
            t_running_loss += period_loss.item()
            phase_running_loss += phase_loss.item()
            amp_running_loss += amp_loss.item()
            mesor_running_loss += mesor_loss.item()

            scores = torch.sigmoid(combined_logits)
            predicted = (scores > 0.5).int()
            all_preds.extend(predicted.detach().cpu().numpy())
            all_scores.extend(scores.detach().to(torch.float).cpu().numpy())

            correct += (predicted == targets.int()).sum().item()
            correct_T += (predicted[:, 0] == targets[:, 0].int()).sum().item()
            correct_Phase += (predicted[:, 1] == targets[:, 1].int()).sum().item()
            correct_Amp += (predicted[:, 2] == targets[:, 2].int()).sum().item()
            correct_Mesor += (predicted[:, 3] == targets[:, 3].int()).sum().item()

            if isinstance(batch_sample_ids, (torch.Tensor, np.ndarray)):
                batch_sample_ids = batch_sample_ids.tolist()
            if isinstance(batch_sample_folders, (torch.Tensor, np.ndarray)):
                batch_sample_folders = batch_sample_folders.tolist()
            
            for sid, sfolder in zip(batch_sample_ids, batch_sample_folders):
                sample_metadata.append({
                    "folder": sfolder,
                    "id": sid
                })

        avg_loss = running_loss / len(self.train_dataloader)
        avg_t_loss = t_running_loss / len(self.train_dataloader)
        avg_phase_loss = phase_running_loss / len(self.train_dataloader)
        avg_amp_loss = amp_running_loss / len(self.train_dataloader)
        avg_mesor_loss = mesor_running_loss / len(self.train_dataloader)

        avg_accuracy = correct / (total * 4)
        t_accuracy = correct_T / total
        phase_accuracy = correct_Phase / total
        amp_accuracy = correct_Amp / total
        mesor_accuracy = correct_Mesor / total

        loss_result = {
            "avg_loss": avg_loss,
            "avg_t_loss": avg_t_loss,
            "avg_amp_loss": avg_amp_loss,
            "avg_phase_loss": avg_phase_loss,
            "avg_mesor_loss": avg_mesor_loss
        }

        accuracy_result = {
            "avg_accuracy": avg_accuracy,
            "t_accuracy": t_accuracy,
            "amp_accuracy": amp_accuracy,
            "phase_accuracy": phase_accuracy,
            "mesor_accuracy": mesor_accuracy
        }

        final_result = {
            "loss": loss_result,
            "accuracy": accuracy_result,
            "sample_metadata": sample_metadata,
            "targets": np.array(all_targets).tolist(),
            "preds": np.array(all_preds).tolist(),
            "scores": np.array(all_scores).tolist(),
            "total_samples": total
        }

        print("Training epoch completed.")
        return final_result

    def evaluate_epoch(self, phase='val'):
        if phase != 'val':
            raise ValueError('Only validation phase is supported (phase=val)')
        dataloader = self.val_dataloader

        print("Starting validation...")
        all_targets, all_preds, all_scores = [], [], []
        sample_metadata = []
        running_loss, t_running_loss, amp_running_loss, phase_running_loss, mesor_running_loss = 0.0, 0.0, 0.0, 0.0, 0.0
        correct, correct_T, correct_Amp, correct_Phase, correct_Mesor = 0, 0, 0, 0, 0
        total = 0

        with torch.no_grad():
            for batch_data_1, input_mask_1, x_marks_1, batch_data_2, input_mask_2, x_marks_2, targets, batch_sample_ids, batch_sample_folders in tqdm(
                self.val_dataloader, total=len(self.val_dataloader), desc="Validating"
            ):
                batch_data_1 = batch_data_1.to(self.device).float()
                batch_data_2 = batch_data_2.to(self.device).float()
                input_mask_1 = input_mask_1.long().to(self.device)
                input_mask_2 = input_mask_2.long().to(self.device)
                x_marks_1 = x_marks_1.to(self.device)
                x_marks_2 = x_marks_2.to(self.device)
                targets = targets.float().to(self.device)

                all_targets.extend(targets.detach().cpu().numpy())
                total += targets.size(0)

                dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8) else torch.float32
                with torch.autocast(device_type=self.device.type, dtype=dtype, enabled=True):
                    output = self.model(
                        x_enc=batch_data_1,
                        input_mask=input_mask_1,
                        x_mark=x_marks_1,
                        x_enc2=batch_data_2,
                        input_mask2=input_mask_2,
                        x_mark2=x_marks_2,
                        reduction=self.args.reduction,
                        return_dict=True
                    )

                    combined_logits = output.logits
                    metadata = output.metadata
                    head1_classes = metadata['head1_classes']
                    head1_logits = combined_logits[:, :head1_classes]
                    head2_logits = combined_logits[:, head1_classes:]

                    head1_targets = targets[:, :head1_classes]
                    head2_targets = targets[:, head1_classes:]

                    loss_head1 = self.criterion_head1(head1_logits, head1_targets)
                    loss_head2 = self.criterion_head2(head2_logits, head2_targets)
                    loss_head1_2 = self.criterion_head1_2(head1_logits, head1_targets)
                    loss_head2_2 = self.criterion_head2_2(head2_logits, head2_targets)
                    total_loss = loss_head1 + loss_head2

                    period_loss = loss_head1_2[:, 0].mean()
                    phase_loss = loss_head1_2[:, 1].mean()
                    amp_loss = loss_head2_2[:, 0].mean()
                    mesor_loss = loss_head2_2[:, 1].mean()

                running_loss += total_loss.item() / 2
                t_running_loss += period_loss.item()
                phase_running_loss += phase_loss.item()
                amp_running_loss += amp_loss.item()
                mesor_running_loss += mesor_loss.item()

                scores = torch.sigmoid(combined_logits)
                predicted = (scores > 0.5).int()
                all_preds.extend(predicted.detach().cpu().numpy())
                all_scores.extend(scores.detach().to(torch.float).cpu().numpy())

                correct += (predicted == targets.int()).sum().item()
                correct_T += (predicted[:, 0] == targets[:, 0].int()).sum().item()
                correct_Phase += (predicted[:, 1] == targets[:, 1].int()).sum().item()
                correct_Amp += (predicted[:, 2] == targets[:, 2].int()).sum().item()
                correct_Mesor += (predicted[:, 3] == targets[:, 3].int()).sum().item()

                if isinstance(batch_sample_ids, (torch.Tensor, np.ndarray)):
                    batch_sample_ids = batch_sample_ids.tolist()
                if isinstance(batch_sample_folders, (torch.Tensor, np.ndarray)):
                    batch_sample_folders = batch_sample_folders.tolist()
                
                for sid, sfolder in zip(batch_sample_ids, batch_sample_folders):
                    sample_metadata.append({
                        "folder": sfolder,
                        "id": sid
                    })

        avg_loss = running_loss / len(self.val_dataloader)
        avg_t_loss = t_running_loss / len(self.val_dataloader)
        avg_phase_loss = phase_running_loss / len(self.val_dataloader)
        avg_amp_loss = amp_running_loss / len(self.val_dataloader)
        avg_mesor_loss = mesor_running_loss / len(self.val_dataloader)

        avg_accuracy = correct / (total * 4)
        t_accuracy = correct_T / total
        phase_accuracy = correct_Phase / total
        amp_accuracy = correct_Amp / total
        mesor_accuracy = correct_Mesor / total

        loss_result = {
            "avg_loss": avg_loss,
            "avg_t_loss": avg_t_loss,
            "avg_amp_loss": avg_amp_loss,
            "avg_phase_loss": avg_phase_loss,
            "avg_mesor_loss": avg_mesor_loss
        }

        accuracy_result = {
            "avg_accuracy": avg_accuracy,
            "t_accuracy": t_accuracy,
            "amp_accuracy": amp_accuracy,
            "phase_accuracy": phase_accuracy,
            "mesor_accuracy": mesor_accuracy
        }

        final_result = {
            "loss": loss_result,
            "accuracy": accuracy_result,
            "sample_metadata": sample_metadata,
            "targets": np.array(all_targets).tolist(),
            "preds": np.array(all_preds).tolist(),
            "scores": np.array(all_scores).tolist(),
            "total_samples": total
        }

        print("Validation completed.")
        return final_result

    def save_checkpoint(self, savePath="best_model.pth", types="test", val_accuracy=None):
        path = os.path.join(self.args.output_path, self.c_time)
        os.makedirs(path, exist_ok=True)
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'epoch': self.epoch,
            'val_accuracy': val_accuracy
        }
        if val_accuracy is not None:
            savePath = savePath.replace(".pth", f"_acc{val_accuracy:.4f}.pth")
        full_checkpoint_path = os.path.join(path, savePath)
        torch.save(checkpoint, full_checkpoint_path)
        print(f"Full checkpoint saved to: {full_checkpoint_path}")
        
        if isinstance(self.model, PeftModel):
            lora_save_path = os.path.join(path, "lora_adapter")
            os.makedirs(lora_save_path, exist_ok=True)
            self.model.save_pretrained(lora_save_path)
            print(f"LoRA adapter saved to: {lora_save_path}")

    def train(self):
        trainSave, valSave = {'loss':[], 'accuracy':[], 'detail':{}}, {'loss':[], 'accuracy':[], 'detail':{}}
        best_val_accuracy = 0.0
        epochs_no_improve = 0

        for epoch in range(self.args.epochs):
            self.epoch = epoch + 1
            train_result = self.train_epoch()
            val_result = self.evaluate_epoch()

            self._log_epoch_result(train_result, val_result)

            trainSave['loss'].append(train_result['loss'])
            trainSave['accuracy'].append(train_result['accuracy'])
            trainSave['detail'] = {k: train_result[k] for k in ("sample_metadata", "targets", "preds", "scores")}
            
            valSave['loss'].append(val_result['loss'])
            valSave['accuracy'].append(val_result['accuracy'])
            valSave['detail'] = {k: val_result[k] for k in ("sample_metadata", "targets", "preds", "scores")}

            train_filename = f"res.json"
            val_filename = f"epoch_{self.epoch:02d}_res.json"
            Metric.save_metrics(trainSave, self.trainAddr, train_filename, self.epoch, self.c_time, mode='w')
            Metric.save_metrics(valSave, self.valAddr, val_filename, self.epoch, self.c_time, mode='w')
            self.save_checkpoint(savePath="current_model.pth", types="current")

            current_val_acc = val_result["accuracy"]['avg_accuracy']
            if current_val_acc > best_val_accuracy + self.min_delta:
                best_val_accuracy = current_val_acc
                epochs_no_improve = 0
                Metric.save_metrics(val_result.copy(), self.valAddr, "best_res.json", self.epoch, self.c_time, 'w')
                self.save_checkpoint(savePath="best_model.pth", val_accuracy=best_val_accuracy)
                print(f"✅ Saved best model (validation accuracy: {best_val_accuracy:.4f})")
            else:
                epochs_no_improve += 1
                print(f"⚠️ Validation accuracy did not improve ({epochs_no_improve}/{self.patience}), current best: {best_val_accuracy:.4f}")

        print("🎉 Training completed!")

    def _log_epoch_result(self, train_result, val_result):
        loss_headers = ["Phase", "Avg Loss", "Period Loss", "Phase Loss", "Amp Loss", "Baseline Loss"]
        loss_data = [
            ["Train", 
             round(train_result['loss']['avg_loss'], 4),
             round(train_result['loss']['avg_t_loss'], 4),
             round(train_result['loss']['avg_phase_loss'], 4),
             round(train_result['loss']['avg_amp_loss'], 4),
             round(train_result['loss']['avg_mesor_loss'], 4)],
            ["Val", 
             round(val_result['loss']['avg_loss'], 4),
             round(val_result['loss']['avg_t_loss'], 4),
             round(val_result['loss']['avg_phase_loss'], 4),
             round(val_result['loss']['avg_amp_loss'], 4),
             round(val_result['loss']['avg_mesor_loss'], 4)]
        ]
        loss_table = tabulate(loss_data, headers=loss_headers, tablefmt="grid", floatfmt=".4f")

        acc_headers = ["Phase", "Avg Acc", "Period Acc", "Phase Acc", "Amp Acc", "Baseline Acc"]
        acc_data = [
            ["Train", 
             round(train_result['accuracy']['avg_accuracy'], 4),
             round(train_result['accuracy']['t_accuracy'], 4),
             round(train_result['accuracy']['phase_accuracy'], 4),
             round(train_result['accuracy']['amp_accuracy'], 4),
             round(train_result['accuracy']['mesor_accuracy'], 4)],
            ["Val", 
             round(val_result['accuracy']['avg_accuracy'], 4),
             round(val_result['accuracy']['t_accuracy'], 4),
             round(val_result['accuracy']['phase_accuracy'], 4),
             round(val_result['accuracy']['amp_accuracy'], 4),
             round(val_result['accuracy']['mesor_accuracy'], 4)]
        ]
        acc_table = tabulate(acc_data, headers=acc_headers, tablefmt="grid", floatfmt=".4f")

        print(f"\n===== Epoch [{self.epoch}/{self.args.epochs}] =====")
        print("Loss results:")
        print(loss_table)
        print("\nAccuracy results:")
        print(acc_table)
        print("=====================================\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--init_lr', type=float, default=1e-6)
    parser.add_argument('--max_lr', type=float, default=1e-4)
    parser.add_argument('--reduction', type=str, default='mean', help='Embedding aggregation method (mean/concat)')
    parser.add_argument('--lora', action='store_true', default=True, help='Enable LoRA during training')
    
    parser.add_argument('--dataset_path', type=str, default="../test/2/", help='Dataset root path')
    parser.add_argument('--pretrained_model_path', type=str, default="saved_nnets/circallm-small/best_model.pth", help='Pretrained model path (file or folder)')
    
    parser.add_argument('--assets_path', type=str, default="assets/test", help='Path to save results')
    parser.add_argument('--output_path', type=str, default="saved_nnets/test", help='Path to save trained models')
    parser.add_argument('--seq_len', type=int, default=72, help='Sequence length per sample (only 72 supported)')
    
    args = parser.parse_args()
    trainer = Circadian_Trainer(args)
    trainer.train()
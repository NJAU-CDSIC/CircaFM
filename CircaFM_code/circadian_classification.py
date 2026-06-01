import argparse
from argparse import Namespace
from email.policy import strict
import random
import numpy as np
import os 
import torch
from data_provider.classfication_datasets import MultipleDataset
from torch.utils.data import DataLoader
from accelerate import Accelerator
from peft import LoraConfig, get_peft_model, PeftModel
from tqdm import tqdm 
from utils.metrics import Metric
from datetime import datetime

class Circadian_Trainer:
    def __init__(self, args: Namespace):
        self.args = args
        self.epoch = 0
        self.trainAddr = os.path.join(self.args.assets_path, "train_results/")
        self.valAddr = os.path.join(self.args.assets_path, "val_results/")
        self.testAddr = os.path.join(self.args.assets_path, "test_results/")

        current_time = datetime.now()
        self.c_time = current_time.strftime("%y_%m_%d_%H_%M")


        # Early stopping parameters
        self.early_stopping_patience = 3
        self.early_stopping_delta = 0.001
        self.best_val_accuracy = 0.0
        self.best_val_loss = float('inf')
        self.counter = 0
        self.early_stop = False

        if hasattr(args, 'early_stopping_patience'):
            self.early_stopping_patience = args.early_stopping_patience
        if hasattr(args, 'early_stopping_delta'):
            self.early_stopping_delta = args.early_stopping_delta
        print("early stop ready!")

        config_dict = {
            "task_name": "classification", 
            "model_name": "CIRCALLM", 
            "transformer_type": "encoder_only", 
            "freeze_embedder": False,
            "freeze_encoder": False,
            "freeze_head": False,
            "learning_rate": 1e-6,
            "num_epochs": 20,
            "n_channels": 1,
            "num_class": 2,
            'reduction': 'mean',
            "d_model": None, 
            "seq_len": 72,
            'enable_gradient_checkpointing': False,
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
                "feed_forward_proj": "gated-gelu",
                "initializer_factor": 1.0,
                "is_encoder_decoder": True,
                "layer_norm_epsilon": 1e-06,
                "model_type": "t5",
                "n_positions": 72,
                "num_decoder_layers": 6,
                "num_heads": 8,
                "num_layers": 6,
                "output_past": True,
                "pad_token_id": 0,
                "relative_attention_max_distance": 128,
                "relative_attention_num_buckets": 32,
                "tie_word_embeddings": False,
                "use_cache": True,
                "vocab_size": 32128
            }
        }

        config = Namespace(**config_dict)
        from models.circaLLM import CIRCALLM
        self.model = CIRCALLM(config)
        print(self.model.head.linear.weight.shape)

        # Load pretrained weights if provided
        pretrained_path = args.pretrained_model_path
        if pretrained_path is not None:
            checkpoint = torch.load(pretrained_path, map_location='cpu')

            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                    print("Detected full training checkpoint (contains 'model_state_dict' key)")
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint

            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

            has_lora = any('lora_' in key for key in state_dict.keys())
            has_peft_prefix = any('base_model.model.' in key for key in state_dict.keys())

            if has_lora or has_peft_prefix:
                print("Detected LoRA / PeftModel weights, will load after adding LoRA structure")
                lora_config = LoraConfig(
                    r=16,
                    lora_alpha=32,
                    target_modules=["q", "v"],
                    lora_dropout=0.1,
                    bias="none",
                    task_type="SEQ_CLS"
                )
                self.model = get_peft_model(self.model, lora_config)
                print("Added LoRA structure to base model")
                load_result = self.model.load_state_dict(state_dict, strict=False)
                print(f"PeftModel weights loaded. Missing keys: {len(load_result.missing_keys)}, unexpected keys: {len(load_result.unexpected_keys)}")
                lora_loaded = any('lora_' in key for key in state_dict.keys() if key not in load_result.missing_keys)
                if not lora_loaded:
                    print("Warning: LoRA parameters may not have been loaded successfully")
            else:
                print("Detected base model weights (no LoRA), will load before adding LoRA")
                load_result = self.model.load_state_dict(state_dict, strict=False)
                print(f"Base model weights loaded. Missing keys: {len(load_result.missing_keys)}, unexpected keys: {len(load_result.unexpected_keys)}")
                lora_config = LoraConfig(
                    r=16,
                    lora_alpha=32,
                    target_modules=["q", "v"],
                    lora_dropout=0.1,
                    bias="none",
                    task_type="SEQ_CLS"
                )
                self.model = get_peft_model(self.model, lora_config)
                print("Added LoRA structure to base model")
        else:
            print("No pretrained weights provided, using randomly initialized base model with LoRA")
            lora_config = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=["q", "v"],
                lora_dropout=0.1,
                bias="none",
                task_type="SEQ_CLS"
            )
            self.model = get_peft_model(self.model, lora_config)
            print("Added LoRA structure to base model")

        # Fix PEFT config issue
        if not hasattr(self.model.config, 'use_return_dict'):
            self.model.config.use_return_dict = True

        self.print_trainable_parameters(self.model)
        self.criterion = torch.nn.CrossEntropyLoss()

        # Initialize circadian classification dataset
        file_paths = self.get_filePaths(self.args.dataset_path)
        self.train_dataset = MultipleDataset(data_split="train", file_paths=file_paths, seq_len=self.args.seq_len, Realcase=True)
        self.val_dataset = MultipleDataset(data_split="val", file_paths=file_paths, seq_len=self.args.seq_len, Realcase=True)
        self.test_dataset = MultipleDataset(data_split="test", file_paths=file_paths, seq_len=self.args.seq_len,  Realcase=True)

        self.train_dataset.labels = self.train_dataset.labels[:,1].astype(int)
        print(self.train_dataset.labels)
        self.val_dataset.labels = self.val_dataset.labels[:,1].astype(int)
        self.test_dataset.labels = self.test_dataset.labels[:,1].astype(int)

        self.train_dataloader = DataLoader(self.train_dataset, batch_size=args.batch_size, shuffle=True)
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=args.batch_size, shuffle=True)
        self.val_dataloader = DataLoader(self.val_dataset, batch_size=args.batch_size, shuffle=True)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.init_lr)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=self.args.max_lr,
                                                             total_steps=self.args.epochs * len(self.train_dataloader))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def check_early_stopping(self, val_accuracy, val_loss):
        """
        Check if early stopping should be triggered.
        """
        if val_accuracy > self.best_val_accuracy + self.early_stopping_delta:
            self.best_val_accuracy = val_accuracy
            self.best_val_loss = val_loss
            self.counter = 0
            print(f"Validation accuracy improved to: {val_accuracy:.4f}, resetting early stopping counter")
            return False
        else:
            self.counter += 1
            print(f"Validation accuracy did not improve, early stopping counter: {self.counter}/{self.early_stopping_patience}")
            if self.counter >= self.early_stopping_patience:
                self.early_stop = True
                print(f"Early stopping triggered! Stopping training at epoch {self.epoch}")
                return True
            return False

    def get_filePaths(self, folder_path) -> list:
        print(os.getcwd())
        originFiles = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
        files = [item for item in originFiles if '(12)' not in item]
        files.sort()
        file_paths = [folder_path + item + "/" for item in files]
        print(file_paths)
        return file_paths

    def train(self):
        trainSave, valSave, testSave = {'loss': [], 'accuracy': [], 'detail': {}}, {'loss': [], 'accuracy': [], 'detail': {}}, {'loss': [], 'accuracy': [], 'detail': {}}
        best_val_accuracy, best_test_accuracy = 0.0, 0.0

        self.early_stop = False
        self.counter = 0
        self.best_val_accuracy = 0.0

        for epoch in range(self.args.epochs):
            if self.early_stop:
                print("Training terminated due to early stopping")
                break

            self.epoch = epoch + 1
            train_result = self.train_epoch()
            val_result = self.evaluate_epoch()

            # Print epoch results
            print(f"Epoch [{self.epoch}/{self.args.epochs}], "
                  f"Train Loss: {train_result['loss']:.4f}, Train Accuracy: {train_result['accuracy']:.4f}, "
                  f"Validation Loss: {val_result['loss']:.4f}, Validation Accuracy: {val_result['accuracy']:.4f}")

            # Save results
            trainSave['loss'].append(train_result['loss'])
            trainSave['accuracy'].append(train_result['accuracy'])
            trainSave['detail'] = {key: train_result[key] for key in ("targets", "preds", "scores")}

            valSave['loss'].append(val_result['loss'])
            valSave['accuracy'].append(val_result['accuracy'])
            valSave['detail'] = {key: val_result[key] for key in ("targets", "preds", "scores")}

            Metric.save_metrics(trainSave, self.trainAddr, "current_train_res.json", self.epoch, str(self.epoch), mode='w')
            Metric.save_metrics(valSave, self.trainAddr, "current_val_res.json", self.epoch, str(self.epoch), mode='w')
            Metric.save_metrics(testSave, self.testAddr, "current_test_res.json", self.epoch, str(self.epoch), mode='w')
            self.save_checkpoint(savePath="current_model.pth", types="current")

            should_stop = self.check_early_stopping(val_result["accuracy"], val_result["loss"])

            if should_stop:
                self.early_stop = True
                print("Early stopping triggered, saving final results...")
                continue

            if val_result["accuracy"] > best_val_accuracy:
                Metric.save_metrics(val_result.copy(), self.valAddr, "best_res.json", self.epoch, mode='w', c_time=self.c_time)
                best_val_accuracy = val_result["accuracy"]
                self.save_checkpoint(savePath="best_model.pth", val_accuracy=best_val_accuracy)
                print(f"Saved best model, validation accuracy: {val_result['accuracy']:.4f} at epoch {self.epoch}")

        print("Training finished!")

        print("Loading best model for final testing...")
        test_result = self.test_epoch()
        print(f"Final test results - Loss: {test_result['loss']:.4f}, Accuracy: {test_result['accuracy']:.4f}")

    def train_epoch(self):
        """
        Train encoder and classification head.
        """
        self.model.to(self.device)
        self.model.train()
        all_targets, all_preds, all_scores = [], [], []
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_data, input_mask, x_marks, targets in tqdm(self.train_dataloader, total=len(self.train_dataloader)):
            self.optimizer.zero_grad()
            batch_data = batch_data.to(self.device).float()
            input_mask, x_marks = input_mask.long().to(self.device), x_marks.to(self.device)
            targets = targets.long().to(self.device)
            all_targets.extend(targets.detach().cpu().numpy())
            total += targets.size(0)

            with torch.autocast(device_type='cuda', dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float32):
                output = self.model(x_enc=batch_data, input_mask=input_mask, x_mark=x_marks, reduction=self.args.reduction)
                logits = output.logits
                loss = self.criterion(logits, targets)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            running_loss += loss.item()

            scores = torch.softmax(logits, dim=1)
            _, predicted = torch.max(scores, 1)
            all_preds.extend(predicted.detach().cpu().numpy())
            correct += (predicted == targets).sum().item()
            all_scores.extend(scores.detach().to(torch.float).cpu().numpy())

        all_targets = np.array(all_targets)
        all_preds = np.array(all_preds)
        all_scores = np.array(all_scores)

        avg_loss = running_loss / len(self.train_dataloader)
        accuracy = correct / total

        result = {
            "loss": avg_loss,
            "accuracy": accuracy,
            "targets": all_targets.tolist(),
            "preds": all_preds.tolist(),
            "scores": all_scores.tolist(),
        }
        return result

    def test_epoch(self):
        return self.evaluate_epoch(phase='test')

    def evaluate_epoch(self, phase='val'):
        if phase == 'val':
            dataloader = self.val_dataloader
        elif phase == 'test':
            dataloader = self.test_dataloader
        else:
            raise ValueError('Invalid phase, please choose val or test')

        self.model.eval()
        self.model.to(self.device)

        all_targets, all_preds, all_scores = [], [], []
        running_loss, correct, total = 0.0, 0, 0

        with torch.no_grad():
            for batch_data, input_mask, x_marks, targets in tqdm(dataloader, total=len(dataloader)):
                batch_data = batch_data.to(self.device).float()
                input_mask, x_marks = input_mask.long().to(self.device), x_marks.to(self.device)
                targets = targets.long().to(self.device)
                total += targets.size(0)
                all_targets.extend(targets.detach().cpu().numpy())

                with torch.autocast(device_type='cuda', dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float32):
                    output = self.model(x_enc=batch_data, input_mask=input_mask, x_mark=x_marks, reduction=self.args.reduction)
                    logits = output.logits
                    loss = self.criterion(logits, targets)
                running_loss += loss.item()

                scores = torch.softmax(logits, dim=1)
                _, predicted = torch.max(scores, 1)
                all_preds.extend(predicted.detach().cpu().numpy())
                correct += (predicted == targets).sum().item()
                all_scores.extend(scores.detach().to(torch.float).cpu().numpy())

        all_targets = np.array(all_targets)
        all_preds = np.array(all_preds)
        all_scores = np.array(all_scores)

        avg_loss = running_loss / len(dataloader)
        accuracy = correct / total

        result = {
            "loss": avg_loss,
            "accuracy": accuracy,
            "targets": all_targets.tolist(),
            "preds": all_preds.tolist(),
            "scores": all_scores.tolist(),
        }
        return result

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

    def print_trainable_parameters(self, model):
        trainable_params = 0
        all_params = 0
        for _, param in model.named_parameters():
            num_params = param.numel()
            all_params += num_params
            if param.requires_grad:
                trainable_params += num_params
        print(f"Trainable parameters: {trainable_params}")
        print(f"Total parameters: {all_params}")
        print(f"Trainable parameter ratio: {100 * trainable_params / all_params:.2f}%")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # training parameters
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--init_lr', type=float, default=1e-6)
    parser.add_argument('--max_lr', type=float, default=1e-4)
    parser.add_argument('--reduction', type=str, default='mean', help='reduction method for circaLLM embeddings, choose from mean or concat')
    parser.add_argument('--lora', action='store_true', default=True, help='enable LoRA')

    # early stopping parameters
    parser.add_argument('--early_stopping_patience', type=int, default=3,
                        help='Number of epochs with no improvement after which training will be stopped')
    parser.add_argument('--early_stopping_delta', type=float, default=0.001,
                        help='Minimum change in the monitored quantity to qualify as an improvement')

    # circadian dataset parameters
    parser.add_argument('--dataset_path', type=str, default="../test/1/", help='path to Circadian dataset')
    parser.add_argument('--pretrained_model_path', type=str, default="saved_nnets/circallm-small/best_model.pth", help='path to load pretrained weight')

    # save address parameters
    parser.add_argument('--save_model_path', type=str, default="pretrained/test", help='path to save model weight')
    parser.add_argument('--assets_path', type=str, default="assets/test", help='path to save important results')
    parser.add_argument('--output_path', type=str, default="saved_nnets/test", help='path to save trained model')
    parser.add_argument('--seq_len', type=int, default=72, help='sequence length for each sample, currently only support 72 for circaLLM')

    args = parser.parse_args()
    trainer = Circadian_Trainer(args)
    trainer.train()
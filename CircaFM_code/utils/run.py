import numpy as np
import torch
from tqdm import tqdm 
#FP32精度
class RUN:
    @staticmethod
    def train_one_epoch(model,train_loader,criterion,optimizer,scheduler,device,reduction="mean"):
        model = model.to(device)

        model.train()
        all_targets = []
        all_preds = []
        all_scores=[]

        running_loss = 0.0
        correct = 0
        total = 0
        i=0
        for data,input_mask,x_marks,targets in train_loader:
            if(i==5):
                break
            i=i+1
            optimizer.zero_grad()
            # forward [batch_size, n_channels, forecast_horizon]
            x_enc=data.to(torch.float32).to(device)
            input_mask=input_mask.long().to(device)
            targets,x_marks=targets.to(device),x_marks.to(device)
            
            
            all_targets.extend(targets.cpu().numpy())

            output = model(x_enc=x_enc,input_mask=input_mask,x_mark=x_marks,reduction=reduction)
            logits=output.logits
            # backward
            loss = criterion(logits, targets)
            
            loss.backward()
            optimizer.step() 
            scheduler.step()

            #loss计算
            running_loss += loss.item()
            #预测标签获取
            scores=torch.softmax(logits,dim=1)
            _, predicted = torch.max(scores, 1)
            all_preds.extend(predicted.detach().cpu().numpy())
            #统计预测正确的标签的个数
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            #计算预测为正类的概率
            all_scores.extend(scores.detach().cpu().numpy())


        #计算关键指标precision,recall,F1-score,auc,AP,mAP
        all_targets = np.array(all_targets)
        all_preds = np.array(all_preds)
        all_scores = np.array(all_scores)
        
        train_loss = running_loss / len(train_loader)
        train_accuracy = correct / total

        result={
            "loss":train_loss,
            "accuracy":train_accuracy,
            "targets":all_targets.tolist(),
            "preds":all_preds.tolist(),
            "scores":all_scores.tolist(),
        }
            # 记录epoch结束的日志
        # tqdm.write(f"Epoch finished: Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.4f}")
        # logging.info(f"Epoch finished: Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.4f}")
        
        return result

    # 定义验证过程
    @staticmethod
    def evaluate(model, dataloader, device, criterion,reduction="reduction"):
        model.eval()
        model.to(device)

        all_targets, all_preds, all_scores = [],[],[]
        running_loss, correct, total = 0.0, 0,0
        with torch.no_grad():
            # i=0
            for batch_data, input_mask,x_marks, targets in tqdm(dataloader, total=len(dataloader)):
                # if i==5:
                #     break
                # i=i+1
                batch_data = batch_data.to(device).float()
                input_mask, x_marks=input_mask.long().to(device), x_marks.to(device)

                targets = targets.to(device)
                total += targets.size(0)
                all_targets.extend(targets.detach().cpu().numpy())

                # with torch.autocast(device_type='cuda', dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float32):
                output = model(x_enc=batch_data,input_mask=input_mask,x_mark=x_marks,reduction=reduction)
                logits=output.logits
                loss = criterion(logits, targets)

                running_loss += loss.item()

                #获取预测标签
                scores=torch.softmax(logits,dim=1)
                _, predicted = torch.max(scores, 1)
                all_preds.extend(predicted.detach().cpu().numpy())
                #统计预测正确的标签的个数
                correct += (predicted == targets).sum().item()
                #计算预测为正类的概率
                all_scores.extend(scores.detach().to(torch.float).cpu().numpy())
        
        #计算关键指标precision,recall,F1-score,auc,AP,mAP
        all_targets = np.array(all_targets)
        all_preds = np.array(all_preds)
        all_scores = np.array(all_scores)

        #计算每个epoch平均loss和accuracy
        avg_loss = running_loss / len(dataloader)
        accuracy = correct / total
        result={
            "loss":avg_loss,
            "accuracy":accuracy,
            "targets":all_targets.tolist(),
            "preds":all_preds.tolist(),
            "scores":all_scores.tolist(),
        }
        return result
    @staticmethod
    def load_checkpoint(model, optimizer, scheduler, load_path):
        checkpoint = torch.load(load_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1  # 从下一个 epoch 开始
        print(f"Checkpoint loaded from {load_path}, resuming from epoch {start_epoch}")
        return model, optimizer, scheduler, start_epoch
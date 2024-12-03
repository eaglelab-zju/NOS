import torch
import torchmetrics
from tqdm import tqdm


def train(model, config, criterion, optimizer, num_epochs, dataloaders):
    valid_best_acc = 0.0
    valid_best_f1 = 0.0
    early_stopping_cnt = 0
    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs-1}")
        print("-" * 10)

        for phase in ["train", "valid"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            total = 0

            test_acc = torchmetrics.Accuracy(task="binary")
            test_recall = torchmetrics.Recall(
                task="binary", average="none", num_classes=2
            )
            test_precision = torchmetrics.Precision(
                task="binary", average="none", num_classes=2
            )
            test_f1 = torchmetrics.F1Score(
                task="binary", average="macro", num_classes=2
            )
            test_auc = torchmetrics.AUROC(task="binary", average="macro", num_classes=2)
            optimizer.zero_grad()
            for idx, batched_graph in enumerate(tqdm(dataloaders[phase])):
                batched_graph = batched_graph.to(config.model_config.device)
                for key in [
                    "images",
                    "labels",
                    "masks",
                    "attributes",
                    "coordinates",
                    "texts",
                ]:
                    batched_graph.ndata[key] = batched_graph.ndata[key].to(
                        config.model_config.device
                    )

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(batched_graph).squeeze(-1)
                    labels, masks = (
                        batched_graph.ndata["labels"],
                        batched_graph.ndata["masks"] > 0,
                    )
                    _, preds = torch.max(outputs, dim=-1)
                    loss = criterion(outputs[masks], labels[masks])

                    if phase == "train":
                        loss.backward()

                        if (
                            config.model_config.accumulation_steps == 0
                            or (idx + 1) % config.model_config.accumulation_steps == 0
                            or idx == len(dataloaders[phase]) - 1
                        ):
                            optimizer.step()
                            optimizer.zero_grad()

                    preds = preds[masks].to("cpu")
                    labels = labels[masks].to("cpu")

                    sz = torch.sum(masks)
                    running_loss += loss.item() * sz
                    running_corrects += torch.sum(preds == labels.data)
                    total += sz

                    epoch_loss = running_loss / total
                    epoch_acc = running_corrects.double() / total

                    with torch.no_grad():
                        outputs = torch.softmax(outputs, dim=-1)
                        outputs = outputs[:, 1][masks].to("cpu")
                    test_acc.update(preds, labels)
                    test_auc.update(outputs, labels)
                    test_recall.update(preds, labels)
                    test_precision.update(preds, labels)
                    test_f1.update(preds, labels)
                    total_acc = test_acc.compute()
                    total_auc = test_auc.compute()
                    total_recall = test_recall.compute()
                    total_precision = test_precision.compute()
                    total_f1 = test_f1.compute()
                    if (
                        config.model_config.accumulation_steps == 0
                        or (idx + 1) % config.model_config.accumulation_steps == 0
                        or idx == len(dataloaders[phase]) - 1
                    ):
                        print(
                            f"\n{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}",
                            end="",
                        )
                        print(
                            f"\nacc: {(100 * total_acc.item()):>0.2f}%, recall: {(100 * total_recall.item()):>0.2f}, precision: {(100 * total_precision.item()):>0.2f}, auc: {(100 * total_auc.item()):>0.2f}, f1: {(100 * total_f1.item()):>0.2f}",
                            end="",
                        )

            epoch_loss = running_loss / total
            epoch_acc = running_corrects.double() / total
            epoch_f1 = total_f1.item()
            print(
                f"\nacc: {(100 * total_acc.item()):>0.2f}%, recall: {(100 * total_recall.item()):>0.2f}, precision: {(100 * total_precision.item()):>0.2f}, auc: {(100 * total_auc.item()):>0.2f}, f1: {(100 * total_f1.item()):>0.2f}",
                end="",
            )
            print(
                f"\n{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} F1: {epoch_f1:.4f}"
            )
            if phase == "valid":
                if epoch_f1 > valid_best_f1:
                    # valid_best_acc = epoch_acc
                    early_stopping_cnt = 0
                    valid_best_f1 = epoch_f1
                    torch.save(model.state_dict(), config.checkpoint_file)
                    print("Saved Best Model!")
                else:
                    early_stopping_cnt += 1
                    if early_stopping_cnt >= config.model_config.early_stopping:
                        print("Reach Early Stopping!")
                        # 清空计算对象
                        test_precision.reset()
                        test_acc.reset()
                        test_recall.reset()
                        test_auc.reset()
                        test_f1.reset()
                        return

            # 清空计算对象
            test_precision.reset()
            test_acc.reset()
            test_recall.reset()
            test_auc.reset()
            test_f1.reset()

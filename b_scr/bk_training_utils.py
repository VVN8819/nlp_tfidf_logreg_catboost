# bk_training_utils.py

"""
Утилиты для обучения Deep Learning моделей.
"""
import torch
import torch.nn as nn
from tqdm import tqdm
# import matplotlib.pyplot as plt
# from IPython.display import clear_output

def train_with_early_stopping(model, train_loader, val_loader,
                              optimizer, loss_fn, num_epochs,
                              patience=3, device='cpu'):
    """
    Универсальный цикл обучения с Early Stopping и визуализацией.
    """
    # пустые списки, чтобы записывать в них результаты после каждой эпохи
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    # Переменные для Early Stopping
    best_val_loss = float('inf')  # Лучший val_loss (начинаем с бесконечности)
    best_model_state = None  # Сохраняем веса лучшей модели
    epochs_without_improvement = 0  # Счетчик эпох без улучшения
    
    print(f" - Начало обучения на устройстве: {str(device).upper()}")
    
    for epoch in range(num_epochs):
        # ====================== TRAIN ============================
        model.train() # Включаем режим обучения (включает Dropout и BatchNorm)
        epoch_loss, epoch_acc = 0.0, 0.0
        
        for input_ids, label_ids in tqdm(train_loader, desc=f'Epoch {epoch+1} [Train]', leave=False):
            input_ids, label_ids = input_ids.to(device), label_ids.to(device)
            
            # 1. обнуляем градиенты
            optimizer.zero_grad()
            
            # 2. Модель делает предсказание
            logits = model(input_ids)
            
            # 3. Сравниваем предсказание модели с правильным ответом и считаем, насколько она ошиблась (Loss)
            loss = loss_fn(logits, label_ids)
            
            # 4. PYTORCH: Вычисляем, в какую сторону нужно изменить веса модели, чтобы ошибка уменьшилась
            loss.backward()
            
            # 5. обновляем веса модели согласно вычислениям выше
            optimizer.step()
            
            # 6. Копим статистику для отчетов в конце эпохи
            epoch_loss += loss.item() * input_ids.size(0)
            
            # Находим, какой класс модель выбрала как наиболее вероятный (argmax)
            preds = logits.argmax(dim=1)
            # Считаем, сколько раз она угадала правильно
            epoch_acc += (preds == label_ids).sum().item()
            
        # считаем СРЕДНИЕ показатели и сохраняем
        train_losses.append(epoch_loss / len(train_loader.dataset))
        train_accs.append(epoch_acc / len(train_loader.dataset))
        
        # ====================== EVAL =============================
        model.eval() # Выключаем режим обучения (отключаем Dropout)
        epoch_loss, epoch_acc = 0.0, 0.0
        
        # torch.inference_mode() говорит PyTorch: "Мы только смотрим, не учимся. 
        with torch.inference_mode():
            for input_ids, label_ids in tqdm(val_loader, desc=f'Epoch {epoch+1} [Val]', leave=False):
                input_ids, label_ids = input_ids.to(device), label_ids.to(device)
                
                # Модель делает предсказание
                logits = model(input_ids)
                # Считаем ошибку
                loss = loss_fn(logits, label_ids)
                
                # Копим статистику
                epoch_loss += loss.item() * input_ids.size(0)
                preds = logits.argmax(dim=1)
                epoch_acc += (preds == label_ids).sum().item()
                
        # Сохранаем средние показатели проверки в "блокнот"
        val_losses.append(epoch_loss / len(val_loader.dataset))
        val_accs.append(epoch_acc / len(val_loader.dataset))
        
        # ================== EARLY STOPPING =======================
        # Берем ошибку на проверке из самой последней (текущей) эпохи
        current_val_loss = val_losses[-1]
        
        # Если текущая ошибка МЕНЬШЕ, чем самая лучшая за все время...
        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            epochs_without_improvement = 0
            
            # сохраняем веса
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f' - Epoch {epoch+1}: Новый рекорд! Val Loss: {current_val_loss:.4f}, Val Acc: {val_accs[-1]:.4f}')
        else:
            # Если рекорда нет, увеличиваем счетчик
            epochs_without_improvement += 1
            print(f' - Epoch {epoch+1}: Нет улучшений ({epochs_without_improvement}/{patience})')
            
        # Если модель не становится лучше уже 'patience' эпох подряд
        if epochs_without_improvement >= patience:
            print(f'\n - Early Stopping сработал на эпохе {epoch+1}.')
            print(f' - Лучшие веса были на эпохе {epoch+1 - patience} с Val Loss: {best_val_loss:.4f}')
            
            # откатываем модель назад во времени к моменту ее максимальной эффективности
            model.load_state_dict(best_model_state)
            break # обучение закончено досрочно
        
    return model, train_losses, val_losses, train_accs, val_accs
    
    
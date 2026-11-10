# ei_train_bilstm_rnn.py

"""
Скрипт для обучения Recurrent neural network (RNN) на текстовых данных.
Использует предобученные эмбеддинги слов через Vocabulary и DataLoader.
"""
import sys
import os
import torch
import torch.nn as nn
import logging
import matplotlib.pyplot as plt

# Добавляем корень проекта в пути
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from b_scr import ba_config
from b_scr.bi_data_loader_dl import load_data_for_dl, create_dataloaders
from b_scr.bj_models_dl import TextBiLSTM
from b_scr.bk_training_utils import train_with_early_stopping

# Настраиваем логирование
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("Обучение Recurrent neural network (RNN)")
    
    # 1. Определяем устройство (GPU или CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"\nИспользуем устройство: {device}")
    
    # 2. Загружаем данные и создаем DataLoader
    logger.info("Загрузка данных и создание DataLoader'ов...")
    train_texts, train_labels, val_texts, val_labels = load_data_for_dl()
    vocabulary, train_loader, val_loader = create_dataloaders(
        train_texts, train_labels,
        val_texts, val_labels,
        batch_size=ba_config.DL_BATCH_SIZE
    )
    
    # 3. Получаем параметры для модели
    vocab_size = len(vocabulary)
    num_classes = len(train_loader.dataset.label_classes)
    logger.info(f"Размер словаря: {vocab_size}")
    logger.info(f"Количество классов: {num_classes}")
    
    # 4. Инициализация модели
    logger.info("\n Инициализация модели TextBiLSTM")
    torch.manual_seed(42)
    
    model = TextBiLSTM(
        vocab_size=vocab_size,
        embedding_dim=ba_config.EMBEDDING_DIM,
        num_classes=num_classes,
        hidden_dim=ba_config.DL_HIDDEN_DIM,
        dropout=ba_config.DL_DROPOUT
    )
    model = model.to(device)
    
    # Подсчитываем количество параметров
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"\nВсего параметров: {total_params:,}")
    logger.info(f"Обучаемых параметров: {trainable_params:,}")
    
    # 5. Оптимизатор и функция потерь
    optimizer = torch.optim.Adam(model.parameters(), lr=ba_config.DL_LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss()
    
    logger.info(f"\nГиперпараметры:")
    logger.info(f" Optimizer: Adam")
    logger.info(f" Learning Rate: {ba_config.DL_LEARNING_RATE}")
    logger.info(f" Batch Size: {ba_config.DL_BATCH_SIZE}")
    logger.info(f" Max Epochs: {ba_config.DL_EPOCHS}")
    logger.info(f" Patience (Early Stopping): 3")
    
    # 6. Запуск обучения
    logger.info("\n Запуск обучения")
    trained_model, train_losses, val_losses, train_accs, val_accs = train_with_early_stopping(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        num_epochs=ba_config.DL_EPOCHS,
        patience=3,
        device=device
    )
    
    # 7. Сохранение модели и артефактов
    logger.info("\n Сохранение модели и артефактов")
    
    os.makedirs(ba_config.MODELS_PATH, exist_ok=True)
    
    # Сохраняем веса модели
    model_save_path = os.path.join(ba_config.MODELS_PATH, "bilstm_model.pt")
    torch.save(trained_model.state_dict(), model_save_path)
    logger.info(f" Веса модели сохранены в: {model_save_path}")
    
    # Сохраняем словарь и маппинг классов
    vocab_save_path = os.path.join(ba_config.MODELS_PATH, "bilstm_vocabulary.pt")
    torch.save({
        'word2idx': vocabulary.word2idx,
        'idx2word': vocabulary.idx2word,
        'label2idx': train_loader.dataset.label2idx,
        'idx2label': train_loader.dataset.idx2label
    }, vocab_save_path)
    logger.info(f" Словарь и маппинг классов сохранены в: {vocab_save_path}")
    
    # 8. Сохраняем графики обучения
    logger.info("\n Сохраняем графики обучения")
    
    os.makedirs(ba_config.PLOTS_PATH, exist_ok=True)
    plot_save_path = os.path.join(ba_config.PLOTS_PATH, "bilstm_training_history.png")
    
    # Создаем фигуру с двумя графиками    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # График 1: Loss
    ax1.plot(train_losses, label='Train Loss', marker='o', markersize=4, linewidth=2)
    ax1.plot(val_losses, label='Val Loss', marker='s', markersize=4, linewidth=2)
    ax1.set_title('Cross-Entropy Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # График 2: Accuracy
    ax2.plot(train_accs, label='Train Acc', marker='o', markersize=4, linewidth=2)
    ax2.plot(val_accs, label='Val Acc', marker='s', markersize=4, linewidth=2)
    ax2.set_title('Classification Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(plot_save_path, dpi=300, bbox_inches='tight')
    logger.info(f" График обучения сохранен в: {plot_save_path}")
    plt.close()
    
    # 9. Финальный отчет
    logger.info("\n Финальный отчет")
    logger.info(f"Лучшая точность на валидации: {max(val_accs):.4f} ({max(val_accs)*100:.2f}%)")
    logger.info(f"Лучший loss на валидации: {min(val_losses):.4f}")
    logger.info(f"Обучение завершено! Модель и графики готовы.")
    
if __name__ == "__main__":
    main()
    
    
    
    
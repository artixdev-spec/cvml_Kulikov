Структура проекта:
- train_model.py — обучение модели, сохранение best_model.pt, classes.json и train.png
- main.py — демонстрация работы модели

Как запускать:
1) Обучение:
python train_model.py --zip_path cyrillic.zip --epochs 15

2) Демонстрация на случайных примерах из датасета:
python main.py --model_path best_model.pt --classes_path classes.json --zip_path cyrillic.zip

3) Демонстрация на отдельном изображении:
python main.py --model_path best_model.pt --classes_path classes.json --image_path path/to/image.png

# Avito service search: candidate generation

Решение задачи candidate generation: для каждого поискового запроса нужно
вернуть до 50 объявлений из корпуса. Целевая метрика — macro Recall@50.

Лучший отправленный результат: **0.64 Recall@50**. Итоговый файл лежит в
[`submissions/answer.csv`](submissions/answer.csv), а воспроизводящий pipeline — в
[`notebooks/03_lora_hybrid_submission.ipynb`](notebooks/03_lora_hybrid_submission.ipynb).

## Идея решения

Один retrieval-метод не покрывал все типы запросов, поэтому использован каскад:

```text
word TF-IDF + char TF-IDF + dense ML-Embed
                         ↓
                candidate union
                         ↓
       category/location/filter-aware reranker
                         ↓
              history fusion + top-50
```

Dense encoder — [`codefuse-ai/ML-Embed-0.6B`](https://huggingface.co/codefuse-ai/ML-Embed-0.6B),
дообученный через LoRA на парах `query → clicked item`. В гибридном reranker
используются reciprocal ranks и cosine каналов, локация, микрокатегория,
рейтинг, число отзывов и мягкий popularity prior.

## Как развивалось решение

| Версия | Leakage-safe validation Recall@50 | Public score |
|---|---:|---:|
| Базовое лексическое решение | — | 0.38 |
| TF-IDF + structure + reranker | 0.5802 | 0.57 |
| Zero-shot dense + lexical hybrid | 0.6323 | 0.62 |
| LoRA dense + lexical hybrid | **0.6484** | **0.64** |

Сравнение сделано на одном group-based split. В `results/` лежат полные
таблицы ablation, а не только лучшие числа.

## Notebook-и

1. [`01_eda_lexical_baseline.ipynb`](notebooks/01_eda_lexical_baseline.ipynb) — EDA,
   leakage-safe split, word/char TF-IDF, структурные признаки и линейный reranker.
2. [`02_zero_shot_dense.ipynb`](notebooks/02_zero_shot_dense.ipynb) — zero-shot ML-Embed,
   exact cosine retrieval и dense top-200.
3. [`03_lora_hybrid_submission.ipynb`](notebooks/03_lora_hybrid_submission.ipynb) —
   **основной submission notebook**: LoRA на 2×T4, dense+lexical fusion, reranker и
   готовый `answer.csv` в одном `Run All`.
4. [`04_hard_negative_hybrid.ipynb`](notebooks/04_hard_negative_hybrid.ipynb) — следующий
   эксперимент: lexical hard negatives, validation loss, early stopping и
   фиксированный предел 1200 optimizer steps. Он не выдаётся за сабмит 0.64.

## Данные и признаки

Исходные данные не включены в Git из-за размера. Notebook-и ищут файлы в Kaggle
Input, а при их отсутствии скачивают [публичный архив](https://disk.yandex.ru/d/sNhfo0YOjGtufg).

Используются:

- текст запроса и текстовые фильтры;
- title, description и parameters объявления;
- category/microcategory и location;
- price, rating и reviews count;
- train-клики как implicit positive feedback.

## Validation

Разбиение сделано по полному query context: text, filters, category, location
и delivery flag. Все clicked benchmark-items для одного context образуют множество
relevant items. Train-пары с validation contexts и positive items из benchmark-корпуса
исключены.

Это не идеальная offline-оценка: клики — неполная implicit-разметка, а отсутствие
клика не доказывает нерелевантность. Поэтому основной целью была устойчивая
полнота, а не точное моделирование CTR.

## Анализ ошибок

| Наблюдаемая ошибка | Изменение в pipeline |
|---|---|
| Разные формулировки одной услуги | multilingual dense retrieval |
| Опечатки и вариативное написание | char n-grams |
| Потеря точных терминов и чисел dense-моделью | word n-grams |
| Объявление из другого города | location features и local quota |
| Фильтры не влияют на поиск | params в query text и отдельный channel |
| Ложные negatives из повторов query context | group filtering и deduplication |
| Низкий recall одного канала | union независимых candidate sets |

## Воспроизводимость

Для полного повторения сабмита:

1. Открыть `notebooks/03_lora_hybrid_submission.ipynb` в Kaggle.
2. Выбрать accelerator **GPU T4 x2** и включить internet.
3. Нажать `Run All`.
4. Скачать `/kaggle/working/answer.csv`.

Обучение не зависит от скорости GPU: используется фиксированная выборка,
одна эпоха и 937 optimizer steps. Версии библиотек закреплены в notebook.
Автоматический batch-size benchmark влияет только на скорость inference, не на порядок
документов или candidate scoring.

Для CPU-экспериментов с lexical baseline:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/jupyter lab notebooks/01_eda_lexical_baseline.ipynb
```

## Структура

```text
notebooks/    эволюция решения и submission notebook
src/          отдельные memory-safe скрипты экспериментов
reports/      EDA, гибриды и обзор открытых материалов Avito
results/      JSON с leakage-safe ablations и коэффициентами reranker
submissions/  лучший отправленный answer.csv
```

## Open-source dependencies

Использованы PyTorch, Transformers, PEFT, scikit-learn, pandas, NumPy, SciPy,
PyArrow и tqdm. Модель и все библиотеки запускаются локально; внешние API
не вызываются.

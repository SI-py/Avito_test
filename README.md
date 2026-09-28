# Avito service search: candidate generation

Финальное решение задачи candidate generation. Для каждого поискового запроса
нужно вернуть 50 `item_id` с максимальным macro Recall@50.

- Финальный notebook: [`avito_solution.ipynb`](avito_solution.ipynb)
- Пример отправленного файла: [`submissions/answer.csv`](submissions/answer.csv)
- Лучший public Recall@50: **0.64**
- SHA256 свежего Kaggle Run All: `954d483842128c516c83adfd24416580242c520e23fce78f76aac4b2d2bcdeba`
- Предыдущие эксперименты, EDA, промежуточные notebook-и и ablations сохранены в отдельной ветке
  [`experiments`](https://github.com/SI-py/Avito_test/tree/experiments)

## Откуда взялась архитектура

Главный ориентир - открытая статья AvitoTech
[«Как работает поисковое ранжирование для миллионов объявлений»](https://habr.com/ru/companies/avito/articles/846832/).
В ней поиск описан как каскад: широкий отбор кандидатов по тексту и фильтрам,
быстрое L1-ранжирование, более дорогое L2 и эвристики. Для этой задачи нужен именно
первый этап, поэтому я оптимизирую не тонкий порядок внутри top-50, а попадание всех
релевантных объявлений в пул кандидатов.

Из статьи я взял три принципа:

1. **Candidate recall ограничивает весь каскад.** Если item не нашёл ни один retrieval-канал,
   reranker его не вернёт. Поэтому используется union независимых lexical, dense и history-каналов.
2. **Lexical и vector search дополняют друг друга.** В материале Avito обратный индекс
   работает вместе с векторным расширением полноты. Здесь ту же роль играют TF-IDF и ML-Embed.
3. **Итоговый score - смесь сигналов.** Avito отдельно учитывает релевантность, вероятность
   сделки и эвристики. В моём reranker аналогично смешиваются retrieval scores, география,
   микрокатегория, rating/reviews и история кликов.

## Что показал EDA

- В train 497 673 строки и 354 241 уникальный полный query context; поэтому validation делится
  по context, а не по строкам.
- 37.5% benchmark-запросов уже встречались в train по нормализованному тексту, но только
  4.45% - вместе с теми же фильтрами и локацией. Query-only history поэтому используется мягко.
- В 83.1% positive pairs локация запроса совпадает с локацией item, но для части микрокатегорий
  эта доля ниже. Поэтому location - признак, а не жёсткий фильтр.
- Только 39.4% clicked pairs покрывают все query tokens в title, но 72.2% - в объединении title, params
  и description. Это мотивирует fielded lexical retrieval и отдельный params-channel.
- 48.6% corpus items входят в группы с одинаковым нормализованным title. Одного title недостаточно:
  нужны description, params, location и признаки качества.

## Как менялось решение

1. **Lexical baseline.** Word/char TF-IDF дал простой memory-safe baseline. Добавление location,
   microcategory, history и линейного reranker подняло validation Recall@50 до 0.5802.
2. **Zero-shot dense channel.** Oracle recall объединённого lexical-пула был около 0.614: ранжировать лучше
   уже недостаточно, нужен новый retrieval-канал. ML-Embed поднял union oracle до 0.672 и итоговый
   Recall@50 до 0.6323.
3. **LoRA domain adaptation.** Модель дообучена на query→clicked-item парах без пересечения с
   validation contexts и benchmark items. Union oracle вырос до 0.6937, а итоговый Recall@50 - до 0.6484.
4. **Lexical hard negatives.** Последний эксперимент увеличил union oracle до 0.7004 и hybrid Recall@50 до
   0.6486, но dense-only Recall@50 снизился до 0.2076, а прирост итоговой метрики оказался всего
   `+0.00018`. Поэтому этот notebook оставлен как research в `experiments`, а в final submission остался более
   простой и устойчивый LoRA hybrid.

## Как запустить

### Kaggle

> **Для точного воспроизведения отправленного `answer.csv` используется Kaggle GPU T4 x2.**

1. Загрузить [`avito_solution.ipynb`](avito_solution.ipynb) в Kaggle.
2. Включить internet и выбрать **GPU**.
3. Выбрать `GPU T4 x2`.
4. Нажать `Run All`.
5. Скачать `/kaggle/working/answer.csv`.

Если три parquet-файла не добавлены в Kaggle Input, notebook сам скачает
[публичный архив задания](https://disk.yandex.ru/d/sNhfo0YOjGtufg).

### Свой Linux-сервер

Нужны Python 3.12, CUDA, PyTorch и одна или две NVIDIA GPU. Положите файлы так:

```text
data/train.parquet
data/benchmark_queries.parquet
data/benchmark_items.parquet
```

Либо укажите каталог через переменную `AVITO_DATA_DIR`. Notebook автоматически
определяет число видимых GPU и запускает `torchrun` с `world_size=1` или `world_size=2`.

## Что делает Run All

Notebook не загружает готовые embeddings, adapters, NPZ или predictions. Он начинает
с трёх исходных parquet-файлов и выполняет весь pipeline:

1. проверка данных и leakage-safe group split;
2. подготовка 30 000 train pairs;
3. LoRA-дообучение `codefuse-ai/ML-Embed-0.6B`;
4. кодирование 189 212 объявлений и всех validation/benchmark queries;
5. exact dense top-200;
6. word/char TF-IDF retrieval;
7. candidate union и category/location/filter-aware reranker;
8. history fusion и строгая проверка `answer.csv`.

```text
word TF-IDF + char TF-IDF + LoRA ML-Embed-0.6B
                         ↓
                candidate union
                         ↓
       category/location/filter-aware reranker
                         ↓
              history fusion + top-50
```

## 1 GPU и 2 GPU

Размер train batch фиксирован: 16 на GPU. Всегда проходит одна полная эпоха:

| Конфигурация | Optimizer steps | Комментарий |
|---|---:|---|
| 1 GPU | 1875 | медленнее; negatives из batch 16 |
| 2 GPU | 937 | быстрее; global negatives из batch 32 |

Для каждого GPU автоматически подбирается безопасный inference batch.

Public score 0.64 получен в режиме 2×T4. Single-GPU mode запускает тот же полный
алгоритм на всех данных, но из-за меньшего global contrastive batch получает другие
LoRA weights и **не предназначен для побитового воспроизведения submission**.

Веса `codefuse-ai/ML-Embed-0.6B` закреплены на revision
`fb82458ca9732a15a9526df89df84d5718efe89f`. В notebook также закреплены seed, версии библиотек,
train sample и число optimizer steps.

Завершённый чистый Kaggle Run All дал `answer.csv` с SHA256
`954d483842128c516c83adfd24416580242c520e23fce78f76aac4b2d2bcdeba`. Этот же hash проверяется
в конце notebook.

## Данные, validation и error analysis

Используются text/filter features запроса, title/description/parameters объявления,
category/microcategory, location, price, rating, reviews count и train-клики.

Разбиение сделано по полному query context. Из train исключены validation contexts
и positive items, присутствующие в benchmark-корпусе.

| Ошибка | Что добавлено |
|---|---|
| Семантические совпадения без общих слов | dense retrieval |
| Опечатки и вариативное написание | char n-grams |
| Потеря точных терминов dense-моделью | word n-grams |
| Объявление из другой локации | location features и local quota |
| Игнорирование фильтров | query params и params channel |
| Низкая полнота одного retrieval-канала | candidate union + reranking |

## Результаты

| Версия | Validation Recall@50 | Public score |
|---|---:|---:|
| TF-IDF + structure | 0.5802 | 0.57 |
| Zero-shot dense hybrid | 0.6323 | 0.62 |
| LoRA dense hybrid | **0.6484** | **0.64** |
| LoRA + lexical hard negatives (research) | **0.6486** | не отправлялся |

Последняя строка - завершённый эксперимент из ветки `experiments`, а не отправленное решение.
Прирост к LoRA hybrid на holdout составил около `+0.00018`, поэтому усложнять final pipeline не стали.

Все библиотеки и модель работают локально; внешние inference API не вызываются.
PyTorch не закреплён в `requirements.txt`, потому что notebook использует CUDA-сборку из
образа Kaggle; остальные прямые зависимости закреплены точными версиями.

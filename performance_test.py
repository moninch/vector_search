import time
import torch
import psutil
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity


# Функция для пуллинга эмбеддингов (среднее значение)
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # Эмбеддинги токенов
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


# Загрузка модели и токенизатора
tokenizer = AutoTokenizer.from_pretrained("ai-forever/sbert_large_mt_nlu_ru")
model = AutoModel.from_pretrained("ai-forever/sbert_large_mt_nlu_ru")

# Сгенерированные заголовки
num_titles = 10000
test_titles = [
    f"Заголовок {i} для теста на производительность" for i in range(num_titles)
]

# 1. Тест токенизации
start_time = time.time()
encoded_input = tokenizer(
    test_titles, padding=True, truncation=True, max_length=128, return_tensors="pt"
)
tokenization_time = time.time() - start_time
print(f"Время токенизации: {tokenization_time:.4f} секунд")

# 2. Тест работы модели
start_time = time.time()
with torch.no_grad():
    model_output = model(**encoded_input)
model_time = time.time() - start_time
print(f"Время работы модели: {model_time:.4f} секунд")

# 3. Тест пуллинга эмбеддингов
start_time = time.time()
sentence_embeddings = mean_pooling(model_output, encoded_input["attention_mask"])
pooling_time = time.time() - start_time
print(f"Время пуллинга эмбеддингов: {pooling_time:.4f} секунд")

# 4. Пропускная способность (Throughput) - сколько заголовков модель обрабатывает за 1 секунду
batch_size = 100  # Размер батча
num_batches = len(test_titles) // batch_size
total_time = 0

for i in range(num_batches):
    batch = test_titles[i * batch_size : (i + 1) * batch_size]
    encoded_input = tokenizer(
        batch, padding=True, truncation=True, max_length=128, return_tensors="pt"
    )
    start_time = time.time()
    with torch.no_grad():
        model_output = model(**encoded_input)
    sentence_embeddings = mean_pooling(model_output, encoded_input["attention_mask"])
    total_time += time.time() - start_time

throughput = len(test_titles) / total_time
print(f"Пропускная способность: {throughput:.2f} заголовков/сек")

# 5. Использование ресурсов CPU и RAM
print(f"Использование CPU: {psutil.cpu_percent()}%")
print(f"Использование памяти: {psutil.virtual_memory().percent}%")

# 6. Использование GPU (если доступен)
if torch.cuda.is_available():
    gpu_mem = torch.cuda.memory_allocated()
    print(f"Использование GPU памяти: {gpu_mem / (1024 ** 2):.2f} MB")
else:
    print("GPU не доступен")

# 7. Масштабируемость (тест на различных размерах данных)
for size in [100, 1000, 10000, 50000]:
    test_titles = [f"Заголовок {i} для теста на масштабируемость" for i in range(size)]
    start_time = time.time()
    encoded_input = tokenizer(
        test_titles, padding=True, truncation=True, max_length=128, return_tensors="pt"
    )
    with torch.no_grad():
        model_output = model(**encoded_input)
    sentence_embeddings = mean_pooling(model_output, encoded_input["attention_mask"])
    total_time = time.time() - start_time
    print(f"Обработка {size} заголовков заняла {total_time:.2f} секунд")

import requests
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np


# Mean Pooling - для получения sentence embeddings
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[
        0
    ]  # Первые выходы модели содержат эмбеддинги токенов
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


# 1. Сбор данных
def scrape_habr_articles():
    base_url = "https://habr.com/ru/all/"
    headers = {"User-Agent": "Mozilla/5.0"}
    articles = []
    # https://habr.com/ru/all/page50/
    for page in range(1, 50):  # Укажите количество страниц для парсинга
        response = requests.get(f"{base_url}page{page}/", headers=headers)
        soup = BeautifulSoup(response.text, "html.parser")
        titles = soup.select("article h2 a")

        for title in titles:
            article_title = title.text.strip()
            article_link = title["href"]
            articles.append((article_title, article_link))

    return articles


# 2. Векторизация статей
def vectorize_articles(articles, tokenizer, model):
    titles = [article[0] for article in articles]
    encoded_input = tokenizer(
        titles, padding=True, truncation=True, max_length=24, return_tensors="pt"
    )

    with torch.no_grad():
        model_output = model(**encoded_input)

    article_embeddings = mean_pooling(model_output, encoded_input["attention_mask"])
    return article_embeddings


# 3. Поиск
def search_articles(query, tokenizer, model, article_embeddings, articles):
    encoded_input = tokenizer(
        query, padding=True, truncation=True, max_length=24, return_tensors="pt"
    )

    with torch.no_grad():
        model_output = model(**encoded_input)

    query_embedding = mean_pooling(model_output, encoded_input["attention_mask"])
    similarities = torch.nn.functional.cosine_similarity(
        query_embedding, article_embeddings
    )
    top_indices = torch.argsort(similarities, descending=True)[:5]  # Топ-5 результатов

    results = []
    for idx in top_indices:
        title, link = articles[idx]
        results.append((title, link, similarities[idx].item()))

    return results


# Основная программа
if __name__ == "__main__":
    print("Собираем статьи с Хабра...")
    articles = scrape_habr_articles()
    print(f"Собрано {len(articles)} статей.")

    print("Загружаем модель Sentence-BERT для русского языка...")
    tokenizer = AutoTokenizer.from_pretrained("ai-forever/sbert_large_mt_nlu_ru")
    model = AutoModel.from_pretrained("ai-forever/sbert_large_mt_nlu_ru")

    print("Векторизуем статьи...")
    article_embeddings = vectorize_articles(articles, tokenizer, model)

    while True:
        query = input("\nВведите запрос для поиска (или 'exit' для выхода): ")
        if query.lower() == "exit":
            break

        print("\nРезультаты поиска:")
        results = search_articles(query, tokenizer, model, article_embeddings, articles)
        for title, link, score in results:
            print(f"Заголовок: {title}\nСсылка: {link}\nСходство: {score:.4f}\n")

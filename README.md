# Ames Housing — Etap 3: LLM Pipeline

Projekt integruje analizę danych nieruchomości (Ames Housing) z modelem językowym
**google/flan-t5-base** do automatycznego generowania komentarzy analitycznych
i odpowiadania na pytania w języku naturalnym.

---

## Struktura projektu

```
.
├── dataset.py                  # Etap 2: wczytanie i transformacja danych
├── analysis.py                 # Etap 2: analizy Q1 (infrastruktura), Q2 (PyTorch), Q3 (Fisher)
├── llm_model.py                # Wrapper FLAN-T5: klasa AmesFlanT5
├── prompt_builder.py           # Konwersja wyników analiz → prompty tekstowe
├── qa_engine.py                # Q&A nad całą bazą ames.db (Text-to-SQL + NL-odpowiedź)
├── main.py                     # Główny pipeline: dane → analizy → prompty → komentarze LLM
├── requirements.txt
├── data/
│   ├── raw/train.csv           # Surowe dane (Kaggle) — lub pobierz z OpenML
│   ├── ames_report.csv         # Tabela analityczna (12 kolumn, ~1460 wierszy)
│   └── ames.db                 # SQLite: tabela 'sales'
├── finetune/
│   ├── prepare_dataset.py      # Generowanie par (prompt, target) → train.json + val.json
│   ├── train.py                # Fine-tuning FLAN-T5-base → flan_t5_ames/final/
│   ├── train.json              # Dane treningowe (generowane)
│   └── val.json                # Dane walidacyjne (generowane)
└── outputs/                    # Wyniki analiz i komentarze LLM
```

---

## Szybki start

### 1. Instalacja zależności

```bash
pip install -r requirements.txt
```

### 2. Fine-tuning modelu (jeden raz)

```bash
# Wygeneruj dane treningowe ze wszystkich analiz
python finetune/prepare_dataset.py

# Uruchom fine-tuning → zapisuje model w flan_t5_ames/final/
python finetune/train.py

# Opcje fine-tuningu:
python finetune/train.py --epochs 5 --batch 4 --lr 3e-4
python finetune/train.py --fp16          # szybszy na GPU (Colab T4)
```

### 3. Główny pipeline

```bash
# Uruchom pełny pipeline (używa fine-tuned modelu automatycznie)
python main.py

# Wymuś model bazowy (bez fine-tuningu)
python main.py --base-model

# Pomiń LLM — tylko analizy i prompty
python main.py --no-llm

# Inference na GPU
python main.py --device cuda
```

### 4. Interaktywny Q&A nad bazą danych

```bash
# Zadaj dowolne pytanie o dane z ames.db
python qa_engine.py

# Przykłady:
#   "Which region has the highest average price?"
#   "How many houses were sold near the railroad?"
#   "What is the average price for apartments with 2 garage spots?"
#   "List top 5 most expensive regions"
```

---

## Jak działa Q&A (qa_engine.py)

```
Pytanie użytkownika
      ↓
FLAN-T5 generuje SQL (Text-to-SQL)
      ↓
SQLite wykonuje zapytanie na ames.db (wszystkie 1460 wierszy, 12 kolumn)
      ↓
FLAN-T5 formułuje odpowiedź w języku naturalnym
      ↓
Odpowiedź + SQL + surowe wiersze
```

Silnik obsługuje **dowolne pytanie**, które można wyrazić jako zapytanie SELECT
na tabeli `sales` — nie tylko podzbiory używane w `prepare_dataset.py`.

---

## Priorytety modelu (main.py i qa_engine.py)

| Kolejność | Warunek | Użyty model |
|-----------|---------|-------------|
| 1 | `--checkpoint <path>` podany | podany checkpoint |
| 2 | `--base-model` flag | `google/flan-t5-base` |
| 3 | `flan_t5_ames/final/` istnieje | fine-tuned model (domyślnie) |
| 4 | nic z powyższych | `google/flan-t5-base` + ostrzeżenie |

---

## Schemat bazy danych (ames.db)

Tabela: `sales` (~1460 wierszy)

| Kolumna | Typ | Opis |
|---------|-----|------|
| Price | REAL | Cena sprzedaży w USD |
| Sales | INTEGER | Liczba transakcji w (Region, Year_Sold) |
| Region | TEXT | Dzielnica (25 unikalnych) |
| Product_Category | TEXT | 'house', 'apartment', 'studio' |
| Discount | TEXT | Sale_Condition\|Sale_Type |
| Discount_Flag | INTEGER | 1 = niestandardowa sprzedaż |
| Price_Elasticity | REAL | Odchylenie od mediany regionalnej |
| Sales_Change_Pct | REAL | Roczna zmiana mediany cen |
| Year_Sold | INTEGER | 2006–2010 |
| Garage_Cars | INTEGER | Miejsca parkingowe (0–4) |
| Condition_1 | TEXT | Sąsiedztwo (Norm, RRAn, RRAe, …) |
| Overall_Qual | INTEGER | Jakość budynku 1–10 |

---

## Wymagania

- Python ≥ 3.10
- ~3–4 GB RAM (inferencja CPU)
- ~6–8 GB RAM (fine-tuning CPU)
- Dysk: ~1.1 GB (wagi modelu) + ~200 MB (checkpoint)

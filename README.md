# Meu voo vai atrasar?

Projeto em Python para treinar um modelo de previsão de atraso de voos e rodar uma interface com Streamlit.

## Como rodar

No PowerShell, execute:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
python src/train_model.py
streamlit run app.py
```

## Pastas principais

- `data/`: base de dados do projeto.
- `src/`: código de treino, predição e utilitários.
- `models/`: arquivos de modelos gerados pelo treinamento.
- `reports/`: relatórios e saídas geradas.
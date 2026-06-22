# Meu voo vai atrasar?

Projeto simples em Python para treinar um modelo de previsao de atraso de voos e rodar uma interface com Streamlit.

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

Os comandos acima tambem estao no arquivo `commands.txt`.

## Pastas principais

- `data/`: base de dados do projeto.
- `src/`: codigo de treino, predicao e utilitarios.
- `models/`: arquivos de modelos gerados pelo treinamento.
- `reports/`: relatorios e saidas geradas.

As pastas `models/` e `reports/` sobem vazias para o Git. Os arquivos gerados dentro delas ficam ignorados.

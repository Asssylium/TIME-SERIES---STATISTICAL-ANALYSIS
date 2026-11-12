param(
    [int]$Port = 8501
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

# uv supplies the project environment without requiring a machine-wide Python
# installation. Existing requirements remain the source of truth for models.
uv run --with-requirements requirements.txt --with streamlit streamlit run app.py --server.port $Port

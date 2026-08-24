# Build the MPIN report. Run from paper/: powershell -File build.ps1
$ErrorActionPreference = 'Stop'
$bin = 'D:\MiKTeX\miktex\bin\x64'
if (Test-Path $bin) { $env:Path = "$bin;$env:Path" }

py -3.13 scripts\gen_tables.py
py -3.13 scripts\gen_figures.py
pdflatex -interaction=nonstopmode -halt-on-error main.tex
biber main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex

Write-Host "`n--- warnings ---"
Select-String -Path main.log -Pattern 'Warning|Undefined' | Select-Object -First 20

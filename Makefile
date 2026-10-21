# Variables
VENV = call-me-maybe
PYTHON = $(VENV)/bin/python
PIP = $(VENV)/bin/pip
PYINSTALLER = $(VENV)/bin/pyinstaller
MAIN = main.py
CONFIG = config.json
FLAKE8_EXCLUDE = $(VENV),mazegenerator.py,.mypy_cache,.mypy.ini,__pycache__,build,dist,*.spec
MYPY_EXCLUDE = '(mazegenerator\.py|build|dist|.*\.spec)'

.PHONY: check_python install run debug clean fclean re lint lint-strict package

check_python:
	@python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' || (echo "Erreur: Python 3.10 ou supérieur est requis pour ce projet." && exit 1)

install: check_python
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install pygame flake8 mypy
	@echo "Installation terminée. Environnement virtuel créé avec succès."

run:
	$(PYTHON) $(MAIN) $(CONFIG)

# Exécution du script en mode débogage avec pdb
debug:
	$(PYTHON) -m pdb $(MAIN) $(CONFIG)

# Suppression des fichiers temporaires et des caches
clean:
	rm -rf __pycache__ .mypy_cache */__pycache__ build dist *.spec

# Nettoyage complet (supprime aussi l'environnement virtuel)
fclean: clean
	rm -rf $(VENV)

# Réinstallation complète
re: fclean install

# Vérification du code (Flake8 et Mypy) avec les flags requis
lint:
	flake8 --exclude=$(FLAKE8_EXCLUDE) .
	mypy --exclude $(MYPY_EXCLUDE) --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs .

# Vérification stricte de la qualité du code
lint-strict:
	flake8 --exclude=$(FLAKE8_EXCLUDE) .
	mypy --exclude $(MYPY_EXCLUDE) --strict .

# Création de l'exécutable du jeu avec PyInstaller
package: install
	$(PIP) install pyinstaller
	$(PYINSTALLER) --noconfirm --windowed --onedir $(MAIN)
	cp config.json dist/pac-man/
	cp -r assets dist/pac-man/
	cp instructions.txt dist/pac-man/
	@echo "Empaquetage terminé ! Le jeu complet se trouve dans le dossier 'dist/'."

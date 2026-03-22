install:
	pip install --upgrade pip && \
	pip install -r requirements.txt
format:
	black backend/*.py frontend/*.py

lint:
	pylint --disable=R,C backend/*.py frontend/*.py

all: install format lint
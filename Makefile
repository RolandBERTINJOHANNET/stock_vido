.PHONY: help install run health
help:
	@echo "make install  - install python deps"
	@echo "make run      - run uvicorn (when app.py exists)"
	@echo "make health   - curl /health"

install:
	pip install -r requirements.txt

run:
	uvicorn service.app:app --reload --port 8080

health:
	curl -s http://localhost:8080/health || true

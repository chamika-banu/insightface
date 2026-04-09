.PHONY: build serve serve-build serve-prod build-prod

# Build

build:
	docker compose build

build-prod:
	docker compose -f docker-compose.yml -f docker-compose.prod.yml build

# Serve

serve:
	docker compose up

serve-build:
	docker compose up --build

serve-prod:
	docker compose -f docker-compose.yml -f docker-compose.prod.yml up

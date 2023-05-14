include .env
service=${AGGREGATOR}
container=${AGGREGATOR}

start:
	docker compose up -d

stop:
	docker compose down

logs:
	docker compose logs

ps:
	docker compose ps

logs_service:
	docker compose logs --follow --no-log-prefix --timestamps $(service)

logs_container:
	docker logs --follow --timestamps $(container)


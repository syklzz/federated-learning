# federated-learning

## Usage

### Run experiment

In order to run experiment please use `make start`.

### View containers

In order to view created containers please use `make ps`.

### View all logs

In order to view logs of all containers please use `make logs`.

### View logs of selected container

In order to view logs of selected container please use ` make logs_container container=<container_name>`, e.g.`make logs_container container=collaborator_2`. Aggregator is the default container. Containers' names are specified in [.env](.env) file.

### Stop experiment

In order to stop experiment please use `make stop`.

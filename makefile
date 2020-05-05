login:
	docker exec -it keras /bin/bash
jp:
	jupyter notebook --ip=0.0.0.0 --port=80  --allow-root
start:
	docker-compose start
stop:
	docker-compose stop

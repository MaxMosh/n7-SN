sudo docker ps --format '{{.Names}}' | xargs -I@ sudo docker exec @ init

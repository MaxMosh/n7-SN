#Here we create a container image where apache2 is installed
FROM ubuntu
RUN apt-get update
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get install -y apache2
RUN apt-get install -y php8.1
RUN apt-get install -y php8.1-intl
RUN apt-get install -y libapache2-mod-php8.0
COPY index.php /var/www/html/
COPY start-apache2.sh /tmp/
CMD /tmp/start-apache2.sh
EXPOSE 80

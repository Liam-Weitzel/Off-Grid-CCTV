FROM ubuntu:16.04

WORKDIR /web

COPY . .

RUN apt-get update && \
apt-get install -y build-essential curl && \
apt-get install -y --no-install-recommends wget unzip git python cmake && \
curl -sL https://deb.nodesource.com/setup_14.x | bash && \
apt-get install -y nodejs && node -v && npm -v && \
rm -rf /var/lib/apt/lists/* && \
npm install && \
apt-get purge -y build-essential curl wget unzip git cmake && \
apt-get autoremove -y --purge && \
npm run build && \
npm install -g serve

CMD ["serve", "-s", "build"]

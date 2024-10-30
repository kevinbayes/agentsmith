FROM rust:1.82.0-slim as builder
WORKDIR /usr/src/myapp
COPY . .
RUN cargo install --path .

FROM ubuntu:24.04

ENV CONFIG_LOCATION "/opt/application-default.json"
ENV RUST_LOG "info"
ENV OPTS ""

MAINTAINER Kevin Bayes "kevin@bayes.me"

RUN rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/cargo/bin/agentsmith-web /usr/local/bin/agentsmith-web
COPY --from=builder /usr/src/myapp/config.json /opt/application-default.json
COPY --from=builder /usr/src/myapp/assets /app
ADD docker/entrypoint.sh /app/entrypoint.sh

RUN chmod u+x /app/entrypoint.sh

EXPOSE 8080

ENTRYPOINT ["./entrypoint.sh"]

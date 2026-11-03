# shorty

A URL shortener in Go. Links are stored in Redis; the admin API speaks gRPC.

Metrics are exported for Prometheus, and the `deploy/` folder has a Helm chart for running it on Kubernetes.

## Roadmap

- Move click events to Kafka (not started)
- Maybe a GraphQL API one day

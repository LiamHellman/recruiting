# order-service

Order management microservice for a fictional bakery chain.

Built with Java 17 and Spring Boot 3. Orders are stored in MySQL and every state change is published to Apache Kafka. Integration tests spin up real databases with Testcontainers.

Build with Maven (`./mvnw verify`). Production runs on AWS ECS behind an Application Load Balancer.

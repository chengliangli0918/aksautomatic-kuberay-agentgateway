FROM golang:1.25.5-bookworm AS build
WORKDIR /src
COPY go.mod go.sum ./
RUN go mod download
COPY cmd/ cmd/
RUN CGO_ENABLED=0 go build -trimpath -ldflags="-s -w" -o /quota-proxy ./cmd/quota-proxy

FROM gcr.io/distroless/static-debian12:nonroot
COPY --from=build /quota-proxy /quota-proxy
USER 65532:65532
EXPOSE 8080
ENTRYPOINT ["/quota-proxy"]

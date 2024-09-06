job "nginx" {
  datacenters = ["dc1"]
  type = "service"

  group "web" {
    count = 1

    task "nginx" {
      driver = "docker"

      config {
        image = "nginx:latest"
        port_map {
          http = 80
        }
      }

      resources {
        network {
          port "http" {
            static = 8080  # Expose on port 8080 on host
          }
        }
      }
    }
  }
}

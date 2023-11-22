# Docker Configuration for Bakllava Application
This directory contains Docker and Docker Compose configurations for running the Bakllava application. It includes multiple services such as a controller, web UI, model worker, and grounded SAM.

## Prerequisites
- Docker: Ensure Docker is installed on your system. Visit Docker's website for installation instructions.
- Docker Compose: Docker Compose is required to run multi-container Docker applications. Install it following the [official guide](https://docs.docker.com/compose/install/).

## Services
The application comprises several services, managed through Docker Compose:

- Controller: Central control service for the application.
- WebUI: Web interface for interacting with the application.
- Model Worker: Handles model processing tasks.
- Grounded SAM: Specialized service for SAM (Segment Anything Model) with Grounding/DINO support.

## Running the Application
Build and Run: Navigate to the root directory of the project and run the following command to build and start all services:


This builds the gsa:v0 image

```bash
cd Grounded-Segment-Anything
make  
```

This down loads the gsa model weights
```bash
chmod +x ./scripts/setup_grounded_sam.sh
./scripts/setup_grounded_sam.sh
```

This creates the bakllava docker network
```bash
docker network create bakllava
```

To build and run the application, use:
```bash
docker compose build --parallel
docker compose up 
```

This command builds images for all services and starts them. The --build flag ensures that Docker Compose rebuilds the images.

## Stopping Services: To stop the services, use:

```bash
docker-compose down
```

## Accessing Services:

- The Controller is accessible at http://localhost:10000.

- The WebUI is available at http://localhost:11000.

- The Model Worker api docs can be viewed at http://localhost:40000/docs 

- Grounded SAM services do not provide web interfaces but can be interacted with via their respective APIs or command line.

GPU Support: For services requiring GPU (like Grounded SAM), ensure your Docker environment is set up with GPU support.

## Configuration
- Docker Compose File: docker-compose.yml in the root directory defines the configuration of each service, including build contexts, Dockerfiles, ports, and dependencies.
- Environment Variables: Set necessary environment variables like HUGGINGFACE_API_TOKEN in your environment or a .env file at the root of the project.
- Customization

You can customize the Docker configurations by editing the Dockerfiles (Dockerfile and ModelWorker.Dockerfile) in the ./docker directory.

Ports and other settings can be adjusted in docker-compose.yml.

## Troubleshooting
- If you encounter issues, check the Docker and Docker Compose logs for details.
- Ensure that all required environment variables are set correctly.
- Verify that your Docker environment supports the necessary capabilities, especially for GPU-accelerated services.
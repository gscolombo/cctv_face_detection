# CCTV Face Detection with Spark Streaming

This project implements an end‑to‑end real‑time face detection and recognition pipeline for CCTV video feeds. It uses **DeepFace** for face detection and embedding extraction, **Apache Spark Structured Streaming** for scalable similarity searches, and **MongoDB** as a vector database of known faces. The entire system is containerized with Docker Compose for easy orchestration and better resource allocation.

## Architecture Overview

The system consists of three main services that communicate via shared storage (Parquet files) and a MongoDB database. The following activity diagram illustrates the data flow and processing steps.

![Activity Diagram](docs/architecture.drawio.png)  


### Core Components

| Service          | Technology                     | Responsibility                                                                 |
|------------------|--------------------------------|--------------------------------------------------------------------------------|
| **Camera Service**   | Python, DeepFace      | Reads video frames, detects and extract faces and writes results as Parquet files. |
| **Processor Service**| PySpark, Spark Structured Streaming | Continuously reads Parquet files, compute embeddings, performs vector similarity search against MongoDB embeddings database, filters by confidence, and outputs matches. |
| **Database**         | MongoDB + mongo-express        | Stores vector embeddings of known faces and recognition logs. |

### High‑Level Pipeline

1. **Video Ingestion & Detection**  
   The `Camera` service reads a video file frame by frame. For each frame, it uses DeepFace’s `yunet` detector (or some other of the available models) to locate and extract faces detected in the frame.

2. **Intermediate Storage**  
   Detected face data (image, bounding box, timestamp) is written as **Parquet files** to a local temporary directory for downstream streaming reads.

3. **Streaming Similarity Search**  
   The `Processor` service uses **Spark Structured Streaming** to monitor the Parquet directory. Each new file is processed as part of a micro‑batch and then deleted. For every face embedding, Spark calls DeepFace’s `search()` function to query the MongoDB vector database.

4. **Confidence Filtering**  
   The system retrieves the most similar known face from MongoDB for every face in the micro-batch, keeping only results above a confidence threshold.

5. **Output**  
   Final matches, if any, are logged or displayed, providing real‑time identification from the CCTV stream.

## Technology Stack

- **Languages & Frameworks**: Python, PySpark, DeepFace  
- **Streaming Engine**: Apache Spark Structured Streaming  
- **Vector Database**: MongoDB (used with DeepFace’s built‑in vector search)  
- **Containerization**: Docker & Docker Compose  
- **Data Format**: Parquet
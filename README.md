---
title: PAIMANA AI - Infrastructure Project Intelligence Platform
emoji: 🏛️
colorFrom: blue
colorTo: orange
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Infrastructure Project Intelligence, Monitoring & Risk Analytics
---

# PAIMANA AI — Infrastructure Project Intelligence Platform

Project Assessment, Intelligence, Monitoring & Analytics Network for Accelerated Infrastructure.

An analytical prototype providing predictive risk intelligence, project monitoring, and evidence-based intervention support for public infrastructure projects.

## Deployment

Configured for **Hugging Face Spaces** using Docker (sdk: docker, port 7860).

### Local Running
`ash
pip install -r requirements.txt
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
`

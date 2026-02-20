"""
Training Strategies for TFT Multi-Component Forecasting.

Five strategies covering different hardware configurations:
    1. cpu_72core  — 72-core CPU server, 148 GB RAM
    2. sagemaker   — Distributed multi-GPU on Amazon SageMaker
    3. gpu_local   — Local single GPU (RTX 3080/3090/4090)
    4. cpu_local   — Local CPU (laptop/desktop)
    5. api_retrain — Incremental re-training via FastAPI service
"""

# =============================================================================
# FastAPI Application for XGBoost Delivery Time Prediction
# Team: Laavanjan 
# Task: API Development
# =============================================================================

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional, Any
import logging
import time
from datetime import datetime
import uvicorn

# Import our inference pipeline
from inference_pipeline import XGBoostInferencePipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="🚚 Delhivery Delivery Time Prediction API",
    description="XGBoost-powered API for predicting delivery times in logistics operations",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instance
model_pipeline = None

# =============================================================================
# Pydantic Models for Request/Response Validation
# =============================================================================


class DeliveryInput(BaseModel):
    """Input model for delivery time prediction"""

    # Core features (required)
    osrm_distance: float = Field(
        ..., description="OSRM calculated distance in meters", ge=0
    )
    osrm_time: float = Field(..., description="OSRM estimated time in seconds", ge=0)
    actual_distance_to_destination: float = Field(
        ..., description="Actual distance to destination in meters", ge=0
    )

    # Optional features with defaults
    cutoff_factor: Optional[float] = Field(1.0, description="Cutoff factor", ge=0)
    factor: Optional[float] = Field(1.0, description="General factor", ge=0)
    time_difference: Optional[float] = Field(
        0.0, description="Time difference in seconds"
    )
    distance_per_min: Optional[float] = Field(0.0, description="Distance per minute")
    planned_duration: Optional[float] = Field(
        0.0, description="Planned duration in seconds", ge=0
    )
    actual_vs_osrm_time: Optional[float] = Field(
        0.0, description="Actual vs OSRM time difference"
    )

    # Boolean features
    is_cutoff: Optional[bool] = Field(False, description="Whether delivery is cut off")
    is_heavy_delay: Optional[bool] = Field(
        False, description="Whether there's heavy delay"
    )

    # Numeric features
    start_scan_to_end_scan: Optional[float] = Field(
        0.0, description="Start to end scan time"
    )
    start_to_cutoff_mins: Optional[float] = Field(
        0.0, description="Start to cutoff minutes"
    )
    center_pair_count: Optional[float] = Field(0.0, description="Center pair count")
    cutoff_hour: Optional[float] = Field(0.0, description="Cutoff hour")
    od_start_time_hour: Optional[float] = Field(
        0.0, description="Origin-destination start time hour"
    )
    cutoff_timestamp_weekday: Optional[float] = Field(
        0.0, description="Cutoff timestamp weekday"
    )

    # Categorical features
    destination_center: Optional[str] = Field(
        "unknown", description="Destination center name"
    )
    destination_name: Optional[str] = Field("unknown", description="Destination name")

    @validator("osrm_distance", "osrm_time", "actual_distance_to_destination")
    def validate_core_features(cls, v):
        if v <= 0:
            raise ValueError("Core features must be positive values")
        return v


class BatchDeliveryInput(BaseModel):
    """Input model for batch predictions"""

    deliveries: List[DeliveryInput] = Field(
        ..., description="List of delivery inputs for batch prediction"
    )

    @validator("deliveries")
    def validate_batch_size(cls, v):
        if len(v) == 0:
            raise ValueError("Batch must contain at least one delivery")
        if len(v) > 100:  # Limit batch size
            raise ValueError("Batch size cannot exceed 100 deliveries")
        return v


class PredictionResponse(BaseModel):
    """Response model for single prediction"""

    predicted_time_seconds: float = Field(
        ..., description="Predicted delivery time in seconds"
    )
    predicted_time_minutes: float = Field(
        ..., description="Predicted delivery time in minutes"
    )
    predicted_time_formatted: str = Field(
        ..., description="Human-readable predicted time (e.g., '2h 30m')"
    )


class SinglePredictionResult(BaseModel):
    """Complete response for single prediction"""

    success: bool = Field(..., description="Whether prediction was successful")
    prediction: Optional[PredictionResponse] = Field(
        None, description="Prediction results"
    )
    model_info: Optional[Dict[str, Any]] = Field(None, description="Model metadata")
    request_id: str = Field(..., description="Unique request identifier")
    timestamp: str = Field(..., description="Request timestamp")
    processing_time_ms: float = Field(
        ..., description="Processing time in milliseconds"
    )


class BatchPredictionResult(BaseModel):
    """Complete response for batch predictions"""

    success: bool = Field(..., description="Whether batch prediction was successful")
    predictions: Optional[List[PredictionResponse]] = Field(
        None, description="List of prediction results"
    )
    model_info: Optional[Dict[str, Any]] = Field(None, description="Model metadata")
    request_id: str = Field(..., description="Unique request identifier")
    timestamp: str = Field(..., description="Request timestamp")
    processing_time_ms: float = Field(
        ..., description="Processing time in milliseconds"
    )
    total_processed: int = Field(..., description="Number of deliveries processed")


class HealthResponse(BaseModel):
    """Health check response"""

    status: str = Field(..., description="Service status")
    timestamp: str = Field(..., description="Health check timestamp")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    uptime_seconds: float = Field(..., description="Service uptime in seconds")


class ErrorResponse(BaseModel):
    """Error response model"""

    success: bool = Field(False, description="Always false for errors")
    error: str = Field(..., description="Error message")
    request_id: str = Field(..., description="Request identifier")
    timestamp: str = Field(..., description="Error timestamp")


# =============================================================================
# Startup and Dependency Functions
# =============================================================================

start_time = time.time()


@app.on_event("startup")
async def startup_event():
    """Initialize the model on startup"""
    global model_pipeline

    logger.info("🚀 Starting Delhivery Delivery Time Prediction API...")

    try:
        model_pipeline = XGBoostInferencePipeline()
        success = model_pipeline.load_model()

        if success:
            logger.info("✅ Model loaded successfully on startup")
        else:
            logger.error("❌ Failed to load model on startup")

    except Exception as e:
        logger.error(f"❌ Startup error: {e}")


def get_model_pipeline():
    """Dependency to get model pipeline"""
    if model_pipeline is None or not model_pipeline.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please check service status.",
        )
    return model_pipeline


def generate_request_id() -> str:
    """Generate unique request ID"""
    return f"req_{int(time.time() * 1000)}"


# =============================================================================
# API Endpoints
# =============================================================================


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    🏥 Health check endpoint for monitoring service status

    Returns:
        HealthResponse: Current service health status
    """
    uptime = time.time() - start_time

    return HealthResponse(
        status=(
            "healthy" if model_pipeline and model_pipeline.is_loaded else "unhealthy"
        ),
        timestamp=datetime.now().isoformat(),
        model_loaded=model_pipeline is not None and model_pipeline.is_loaded,
        uptime_seconds=uptime,
    )


@app.post("/predict", response_model=SinglePredictionResult, tags=["Prediction"])
async def predict_delivery_time(
    delivery_input: DeliveryInput,
    pipeline: XGBoostInferencePipeline = Depends(get_model_pipeline),
):
    """
    🎯 Predict delivery time for a single delivery

    This endpoint takes delivery parameters and returns the predicted delivery time
    using the trained XGBoost model.

    Args:
        delivery_input: Delivery parameters including distances, times, and other features

    Returns:
        SinglePredictionResult: Prediction results with metadata
    """
    request_id = generate_request_id()
    start_time_req = time.time()

    try:
        logger.info(f"📊 Processing prediction request: {request_id}")

        # Convert Pydantic model to dict
        input_data = delivery_input.dict()

        # Make prediction
        result = pipeline.predict(input_data)

        if result["success"]:
            processing_time = (time.time() - start_time_req) * 1000

            response = SinglePredictionResult(
                success=True,
                prediction=PredictionResponse(**result["predictions"]),
                model_info=result["model_info"],
                request_id=request_id,
                timestamp=datetime.now().isoformat(),
                processing_time_ms=processing_time,
            )

            logger.info(
                f"✅ Prediction successful: {request_id} - {result['predictions']['predicted_time_formatted']}"
            )
            return response
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail=result["error"]
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Prediction error for {request_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}",
        )


@app.post("/predict/batch", response_model=BatchPredictionResult, tags=["Prediction"])
async def predict_batch_delivery_times(
    batch_input: BatchDeliveryInput,
    pipeline: XGBoostInferencePipeline = Depends(get_model_pipeline),
):
    """
    📦 Predict delivery times for multiple deliveries in batch

    This endpoint processes multiple delivery predictions in a single request
    for improved efficiency when dealing with multiple deliveries.

    Args:
        batch_input: List of delivery parameters

    Returns:
        BatchPredictionResult: Batch prediction results with metadata
    """
    request_id = generate_request_id()
    start_time_req = time.time()

    try:
        logger.info(
            f"📊 Processing batch prediction request: {request_id} - {len(batch_input.deliveries)} deliveries"
        )

        # Convert Pydantic models to list of dicts
        input_data = [delivery.dict() for delivery in batch_input.deliveries]

        # Make batch prediction
        result = pipeline.predict(input_data)

        if result["success"]:
            processing_time = (time.time() - start_time_req) * 1000

            predictions = [PredictionResponse(**pred) for pred in result["predictions"]]

            response = BatchPredictionResult(
                success=True,
                predictions=predictions,
                model_info=result["model_info"],
                request_id=request_id,
                timestamp=datetime.now().isoformat(),
                processing_time_ms=processing_time,
                total_processed=len(predictions),
            )

            logger.info(
                f"✅ Batch prediction successful: {request_id} - {len(predictions)} deliveries processed"
            )
            return response
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail=result["error"]
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Batch prediction error for {request_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}",
        )


@app.get("/model/info", tags=["Model"])
async def get_model_info(
    pipeline: XGBoostInferencePipeline = Depends(get_model_pipeline),
):
    """
    ℹ️ Get information about the loaded model

    Returns:
        Dict: Model information and statistics
    """
    return {
        "model_type": "XGBoost",
        "version": "1.0.0",
        "features_count": len(pipeline.expected_features),
        "features": pipeline.expected_features,
        "model_loaded": pipeline.is_loaded,
        "model_path": pipeline.model_path,
    }


@app.get("/", tags=["General"])
async def root():
    """
    🏠 Root endpoint with API information
    """
    return {
        "message": "🚚 Delhivery Delivery Time Prediction API",
        "version": "1.0.0",
        "team": "Laavanjan",
        "docs": "/docs",
        "health": "/health",
        "model_info": "/model/info",
    }


# =============================================================================
# Exception Handlers
# =============================================================================


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            request_id=generate_request_id(),
            timestamp=datetime.now().isoformat(),
        ).dict(),
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions"""
    logger.error(f"❌ Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            request_id=generate_request_id(),
            timestamp=datetime.now().isoformat(),
        ).dict(),
    )


# =============================================================================
# Main Application Runner
# =============================================================================

if __name__ == "__main__":
    # Run the application
    uvicorn.run("api_app:app", host="0.0.0.0", port=8000, reload=True, log_level="info")

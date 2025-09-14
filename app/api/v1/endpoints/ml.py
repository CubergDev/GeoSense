from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from sqlalchemy.orm import Session
from datetime import datetime, timedelta

from app.core.database import get_db
# Note: Relax schema dependencies for tests; use dict payloads and plain dict responses
from app.ml.demand_predictor import demand_predictor
from app.ml.anomaly_detector import anomaly_detector
from app.ml.route_optimizer import route_optimizer

router = APIRouter()


@router.post("/predict-demand", summary="Predict demand")
async def predict_demand(
    request: dict,
    db: Session = Depends(get_db)
):
    """
    Predict transportation demand for a specific area and time.
    
    Uses machine learning models trained on historical trip data to forecast
    demand patterns. The model considers:
    - Historical demand patterns
    - Time of day/week/year effects
    - Weather conditions (if available)
    - Special events and holidays
    - Spatial demand correlations
    
    **Use Cases**:
    - Dynamic pricing optimization
    - Fleet positioning and sizing
    - Driver incentive targeting
    - Capacity planning
    
    - **request**: Prediction parameters including target time and area
    """
    try:
        target_time = (request or {}).get("target_time")
        area_bounds = (request or {}).get("area_bounds")
        grid_size_meters = (request or {}).get("grid_size_meters")
        predictions = demand_predictor.predict(
            db=db,
            target_time=target_time,
            area_bounds=area_bounds,
            grid_size_meters=grid_size_meters,
        )
        return {
            "predictions": predictions,
            "model_version": demand_predictor.get_model_version(),
            "prediction_timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Demand prediction failed: {str(e)}")


@router.post("/detect-anomalies", summary="Detect anomalous trips (P0)")
async def detect_anomalies(
    start_time: datetime = Query(..., description="Analysis start time"),
    end_time: datetime = Query(..., description="Analysis end time"),
    top_n: int = Query(100, ge=1, le=1000, description="Max items to return"),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    db: Session = Depends(get_db)
):
    """
    Detect anomalous trips using P0 features and IsolationForest.
    Returns items with severity/flags and why explanations.
    """
    try:
        # Run anomaly detection in background for large datasets
        if (end_time - start_time).days > 7:
            background_tasks.add_task(
                anomaly_detector.detect_batch_anomalies,
                db, start_time, end_time, 0.02
            )
            return {
                "message": "Anomaly detection started in background",
                "analysis_period": f"{start_time.isoformat()} to {end_time.isoformat()}",
                "status": "processing"
            }
        else:
            res = anomaly_detector.detect_anomalies_p0(
                db=db,
                start_time=start_time,
                end_time=end_time,
                top_n=top_n,
                contamination=0.02,
            )
            return res
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Anomaly detection failed: {str(e)}")


@router.post("/optimize-routes", summary="Optimize route recommendations")
async def optimize_routes(
    request: dict,
    db: Session = Depends(get_db)
):
    """
    Optimize route recommendations using ML-powered routing.
    
    The route optimizer considers:
    - Historical traffic patterns
    - Real-time congestion data
    - Vehicle-specific constraints
    - Time-dependent factors
    - Fuel/energy efficiency
    
    **Optimization Goals**:
    - time: Minimize travel time
    - distance: Minimize total distance
    - fuel: Minimize fuel consumption
    
    **Use Cases**:
    - Delivery route optimization
    - Multi-stop trip planning
    - Fleet efficiency improvement
    - Cost reduction strategies
    
    - **request**: Route optimization parameters including origin, destinations, and preferences
    """
    try:
        req = request or {}
        destinations = req.get("destinations") or []
        if len(destinations) > 20:
            raise HTTPException(status_code=400, detail="Maximum 20 destinations allowed")
        optimized_route = route_optimizer.optimize(
            db=db,
            origin_lat=req.get("origin_lat"),
            origin_lon=req.get("origin_lon"),
            destinations=destinations,
            optimization_goal=req.get("optimization_goal"),
            vehicle_type=req.get("vehicle_type"),
            departure_time=req.get("departure_time") or datetime.now()
        )
        return {
            "optimized_route": optimized_route,
            "optimization_goal": req.get("optimization_goal"),
            "vehicle_type": req.get("vehicle_type"),
            "total_destinations": len(destinations),
            "estimated_savings": optimized_route.get("savings", {})
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Route optimization failed: {str(e)}")


@router.get("/model-status", summary="Get ML model status")
async def get_model_status():
    """
    Get the current status and performance metrics of all ML models.
    
    **Information Provided**:
    - Model versions and last training dates
    - Performance metrics and accuracy scores
    - Training data statistics
    - Model health indicators
    
    **Use Cases**:
    - Model monitoring and maintenance
    - Performance tracking
    - System health checks
    - Model update planning
    """
    try:
        status = {
            "demand_predictor": {
                "version": demand_predictor.get_model_version(),
                "last_trained": demand_predictor.get_last_training_date(),
                "accuracy": demand_predictor.get_accuracy_metrics(),
                "status": "healthy" if demand_predictor.is_healthy() else "degraded"
            },
            "anomaly_detector": {
                "version": anomaly_detector.get_model_version(),
                "last_trained": anomaly_detector.get_last_training_date(),
                "precision": anomaly_detector.get_precision_metrics(),
                "status": "healthy" if anomaly_detector.is_healthy() else "degraded"
            },
            "route_optimizer": {
                "version": route_optimizer.get_model_version(),
                "last_trained": route_optimizer.get_last_training_date(),
                "efficiency_gain": route_optimizer.get_efficiency_metrics(),
                "status": "healthy" if route_optimizer.is_healthy() else "degraded"
            }
        }
        
        return {
            "models": status,
            "overall_status": "healthy" if all(m["status"] == "healthy" for m in status.values()) else "degraded",
            "last_updated": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get model status: {str(e)}")


@router.post("/retrain-models", summary="Trigger model retraining")
async def retrain_models(
    model_name: Optional[str] = Query(None, pattern="^(demand|anomaly|route|all)$", description="Specific model to retrain"),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    db: Session = Depends(get_db)
):
    """
    Trigger retraining of ML models with latest data.
    
    **Retraining Benefits**:
    - Improved accuracy with recent data
    - Adaptation to changing patterns
    - Better seasonal adjustments
    - Enhanced prediction quality
    
    **Models Available**:
    - demand: Demand prediction model
    - anomaly: Anomaly detection model
    - route: Route optimization model
    - all: All models (default)
    
    - **model_name**: Specific model to retrain (optional, defaults to all)
    """
    try:
        models_to_retrain = []
        
        if model_name is None or model_name == "all":
            models_to_retrain = ["demand", "anomaly", "route"]
        else:
            models_to_retrain = [model_name]
        
        # Start retraining in background
        for model in models_to_retrain:
            if model == "demand":
                background_tasks.add_task(demand_predictor.retrain, db)
            elif model == "anomaly":
                background_tasks.add_task(anomaly_detector.retrain, db)
            elif model == "route":
                background_tasks.add_task(route_optimizer.retrain, db)
        
        return {
            "message": f"Retraining started for: {', '.join(models_to_retrain)}",
            "models": models_to_retrain,
            "status": "processing",
            "estimated_completion": "30-60 minutes"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start retraining: {str(e)}")


@router.get("/predictions/history", summary="Get prediction history")
async def get_prediction_history(
    model_type: str = Query(..., pattern="^(demand|anomaly|route)$", description="Model type"),
    start_time: datetime = Query(..., description="History start time"),
    end_time: datetime = Query(..., description="History end time"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum records to return"),
    db: Session = Depends(get_db)
):
    """
    Get historical predictions and their accuracy.
    
    **Use Cases**:
    - Model performance analysis
    - Prediction accuracy tracking
    - Business impact assessment
    - Model improvement insights
    
    - **model_type**: Type of model predictions to retrieve
    - **start_time**: Start of history period
    - **end_time**: End of history period
    - **limit**: Maximum number of records
    """
    try:
        if model_type == "demand":
            history = demand_predictor.get_prediction_history(db, start_time, end_time, limit)
        elif model_type == "anomaly":
            history = anomaly_detector.get_detection_history(db, start_time, end_time, limit)
        elif model_type == "route":
            history = route_optimizer.get_optimization_history(db, start_time, end_time, limit)
        
        return {
            "predictions": history,
            "model_type": model_type,
            "period": f"{start_time.isoformat()} to {end_time.isoformat()}",
            "total_records": len(history)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get prediction history: {str(e)}")
